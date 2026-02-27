#include "acrelab-gauss/arrow_parquet_writer.hpp"
#include "batch_job.hpp"

namespace acrelab {

// Explicit template instantiation for BatchJob
template void ArrowParquetWriter::WriteBatchJob<BatchJob>(const BatchJob* job);

} // namespace acrelab