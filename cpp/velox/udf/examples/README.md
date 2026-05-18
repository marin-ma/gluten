# Velox UDF Examples

This directory contains example shared libraries for Gluten Velox UDF loading:

- `myudf` from `MyUDF.cc`
- `myudaf` from `MyUDAF.cc`
- `sessiontimeout` from `SessionTimeout.cc` (window function)

## SessionTimeout UDWF

`SessionTimeout.cc` registers:

- function name: `test.org.apache.spark.sql.window.SessionTimeout`
- signatures:
  - `(bigint, integer) -> bigint`
  - `(bigint, bigint) -> bigint`
  - `(timestamp, integer) -> bigint`
  - `(timestamp, bigint) -> bigint`

The implementation emits a monotonically increasing session id within each partition based on the timestamp gap threshold.

## Build

Adjust the build dir if your local path differs.

```bash
cmake --build /Users/rong/workspace/github/apache/gluten/cpp/build --target sessiontimeout -j
```

## Notes

- This plugin exports window entry symbols (`getNumWindow`, `getWindowEntries`) and also exports `registerUdf` to match current `UdfLoader::registerUdf()` behavior.
- The function name used in native registration must match the name resolved by JVM-side UDWF mapping.

