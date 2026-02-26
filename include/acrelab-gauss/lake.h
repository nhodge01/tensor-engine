#pragma once

#include <duckdb.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

struct TextBatch {
  std::vector<uint64_t> gids;
  std::vector<std::string> texts;
  size_t num_rows = 0;
};

class DuckLake {
private:
  duckdb::DuckDB db;
  duckdb::Connection conn;

  using clock = std::chrono::high_resolution_clock;

public:
  DuckLake();
  //   const std::string& endpoint,
  //   const std::string& access_key,
  //   const std::string& secret_key,

  //   const std::string& db_host,
  //   const std::string& db_name,
  //   const std::string& db_user,
  //   const std::string& db_pass,
  //   const std::string& db_port
  // );

  TextBatch extract_text_batch(const std::string& query, size_t reserve_size);

};
