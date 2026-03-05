#ifndef ACRELAB_GAUSS_ARROW_PARQUET_WRITER_HPP
#define ACRELAB_GAUSS_ARROW_PARQUET_WRITER_HPP

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/filesystem/s3fs.h>
#include <arrow/filesystem/api.h>
#include <parquet/arrow/writer.h>

namespace acrelab {

struct S3Config {
    std::string endpoint;
    std::string access_key;
    std::string secret_key;
    std::string bucket_path;
    bool use_ssl = false;
};

class ArrowParquetWriter {
public:
    /**
     * @brief Construct writer with dynamic multi-vector schema
     */
    explicit ArrowParquetWriter(const S3Config& config, int embedding_dim, const std::vector<std::string>& column_names)
        : config_(config), embedding_dim_(embedding_dim), column_names_(column_names) {

        auto status = arrow::fs::EnsureS3Initialized();
        if (!status.ok()) {
            throw std::runtime_error("Failed to initialize S3: " + status.ToString());
        }

        arrow::fs::S3Options s3_options;
        s3_options.endpoint_override = config.endpoint;
        s3_options.scheme = config.use_ssl ? "https" : "http";
        s3_options.ConfigureAccessKey(config.access_key, config.secret_key);

        auto fs_result = arrow::fs::S3FileSystem::Make(s3_options);
        if (!fs_result.ok()) {
            throw std::runtime_error("Failed to create S3 filesystem: " + fs_result.status().ToString());
        }
        s3_fs_ = fs_result.ValueOrDie();

        // --- DYNAMIC SCHEMA BUILDING ---
        std::vector<std::shared_ptr<arrow::Field>> fields;
        fields.push_back(arrow::field("gid", arrow::int64()));

        // Add a vector column for every entry in column_names (e.g., v1, v2, v3...)
        for (const auto& col_name : column_names_) {
            fields.push_back(arrow::field(col_name, arrow::fixed_size_list(arrow::float32(), embedding_dim_)));
        }
        schema_ = arrow::schema(fields);
    }

    void Open() {
        auto stream_result = s3_fs_->OpenOutputStream(config_.bucket_path);
        if (!stream_result.ok()) {
            throw std::runtime_error("Failed to open S3 output stream: " + stream_result.status().ToString());
        }
        output_stream_ = stream_result.ValueOrDie();

        parquet::WriterProperties::Builder props_builder;
        props_builder.compression(parquet::Compression::SNAPPY);
        props_builder.version(parquet::ParquetVersion::PARQUET_2_6);
        auto writer_props = props_builder.build();

        auto writer_status = parquet::arrow::FileWriter::Open(
            *schema_, arrow::default_memory_pool(), output_stream_, writer_props
        );

        if (!writer_status.ok()) {
            throw std::runtime_error("Failed to create Parquet writer: " + writer_status.status().ToString());
        }

        writer_ = std::move(writer_status.ValueOrDie());
        is_open_ = true;
        total_rows_written_ = 0;
        std::cout << "ArrowParquetWriter: Opened Multi-Vector S3 stream (" << column_names_.size() << " vectors/row)" << std::endl;
    }

    /**
     * @brief Multi-Vector WriteBatchJob
     * Unrolls the interleaved GPU embeddings into the specific Arrow columns.
     */
    template<typename T_BatchJob>
    void WriteBatchJob(const T_BatchJob* job) {
        if (!is_open_) throw std::runtime_error("Writer not open.");
        if (!job || job->valid_items <= 0) return;

        int num_rows = job->valid_items;
        int num_v = column_names_.size();

        // 1. GID Column (Zero-Copy)
        auto gid_buffer = arrow::Buffer::Wrap(job->row_ids.data(), num_rows * sizeof(int64_t));
        auto gid_array = std::make_shared<arrow::Int64Array>(num_rows, gid_buffer);

        std::vector<std::shared_ptr<arrow::Array>> columns;
        columns.push_back(gid_array);

        // 2. Vector Columns
        // Because the GPU buffer is [Row0_V1, Row0_V2... Row0_VN, Row1_V1...],
        // we extract each version into its own contiguous Arrow array.
        for (int v = 0; v < num_v; ++v) {
            auto builder = std::make_shared<arrow::FixedSizeListBuilder>(
                arrow::default_memory_pool(),
                std::make_shared<arrow::FloatBuilder>(),
                arrow::fixed_size_list(arrow::float32(), embedding_dim_)
            );

            auto* value_builder = static_cast<arrow::FloatBuilder*>(builder->value_builder());

            for (int r = 0; r < num_rows; ++r) {
                // Pointer math to find the specific version for this row
                float* src = &job->pinned_embeddings[(r * num_v + v) * embedding_dim_];
                (void)value_builder->AppendValues(src, embedding_dim_);
                (void)builder->Append();
            }

            std::shared_ptr<arrow::Array> vec_col;
            (void)builder->Finish(&vec_col);
            columns.push_back(vec_col);
        }

        auto batch = arrow::RecordBatch::Make(schema_, num_rows, columns);
        auto status = writer_->WriteRecordBatch(*batch);
        if (!status.ok()) throw std::runtime_error("Parquet Write Error: " + status.ToString());

        total_rows_written_ += num_rows;
    }

    void Close() {
        if (!is_open_) return;
        writer_->Close();
        output_stream_->Close();
        is_open_ = false;
    }

    ~ArrowParquetWriter() {
        if (is_open_) Close();
        arrow::fs::FinalizeS3();
    }

    size_t GetTotalRowsWritten() const { return total_rows_written_; }

private:
    S3Config config_;
    int embedding_dim_;
    std::vector<std::string> column_names_;
    std::shared_ptr<arrow::Schema> schema_;
    std::shared_ptr<arrow::fs::S3FileSystem> s3_fs_;
    std::shared_ptr<arrow::io::OutputStream> output_stream_;
    std::unique_ptr<parquet::arrow::FileWriter> writer_;
    bool is_open_ = false;
    size_t total_rows_written_ = 0;
};

inline std::unique_ptr<ArrowParquetWriter> CreateS3WriterFromEnv(
    const std::string& output_path,
    int embedding_dim,
    const std::vector<std::string>& column_names
) {
    S3Config config;
    config.endpoint = std::getenv("MINIO_ENDPOINT");
    config.access_key = std::getenv("MINIO_ACCESS_KEY");
    config.secret_key = std::getenv("MINIO_SECRET_KEY");
    config.bucket_path = output_path;
    config.use_ssl = false;

    return std::make_unique<ArrowParquetWriter>(config, embedding_dim, column_names);
}

} // namespace acrelab

#endif
