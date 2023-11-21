/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.glutenproject.backendsapi.velox

import io.glutenproject.backendsapi.TestApi
import io.glutenproject.backendsapi.TestApi.{SparkVersion, SparkVersion32, SparkVersion33}

class TestApiImpl extends TestApi {

  /**
   * List of supported cases to run with Velox backend, in lower case. Please add to the supported
   * list after enabling a sql test.
   */
  override def getSupportedSQLQueryTests(sparkVersion: SparkVersion): Set[String] = {
    sparkVersion match {
      case SparkVersion32() =>
        SUPPORTED_SQL_QUERY_LIST_SPARK32
      case SparkVersion33() =>
        SUPPORTED_SQL_QUERY_LIST_SPARK33
    }
  }

  override def getOverwriteSQLQueryTests(sparkVersion: SparkVersion): Set[String] = {
    sparkVersion match {
      case SparkVersion32() =>
        OVERWRITE_SQL_QUERY_LIST_SPARK32
      case SparkVersion33() =>
        OVERWRITE_SQL_QUERY_LIST_SPARK33
    }
  }

  private val SUPPORTED_SQL_QUERY_LIST_SPARK32: Set[String] =
    Set(
      "bitwise.sql",
      "cast.sql",
      "change-column.sql",
      "charvarchar.sql",
      "columnresolution-negative.sql",
      "columnresolution-views.sql",
      "columnresolution.sql",
      "comments.sql",
      "comparator.sql",
      "count.sql",
      "cross-join.sql",
      "csv-functions.sql",
      "cte-legacy.sql",
      "cte-nested.sql",
      "cte-nonlegacy.sql",
      "cte.sql",
      "current_database_catalog.sql",
      "date.sql",
      "datetime-formatting-invalid.sql",
      "datetime-formatting-legacy.sql",
      "datetime-formatting.sql",
      "datetime-legacy.sql",
      "datetime-parsing-invalid.sql",
      "datetime-parsing-legacy.sql",
      "datetime-parsing.sql",
      "datetime-special.sql",
      "decimalArithmeticOperations.sql",
      "describe-part-after-analyze.sql",
      "describe-query.sql",
      "describe-table-after-alter-table.sql",
      // result match, but the order is not right
      // "describe-table-column.sql",
      "describe.sql",
      "except-all.sql",
      "except.sql",
      "extract.sql",
      "group-by-filter.sql",
      "group-by-ordinal.sql",
      "grouping_set.sql",
      "having.sql",
      "ignored.sql",
      "inline-table.sql",
      "inner-join.sql",
      "intersect-all.sql",
      "interval.sql",
      "join-empty-relation.sql",
      "join-lateral.sql",
      "json-functions.sql",
      "like-all.sql",
      "like-any.sql",
      "limit.sql",
      "literals.sql",
      "map.sql",
      "misc-functions.sql",
      "natural-join.sql",
      "null-handling.sql",
      "null-propagation.sql",
      "operators.sql",
      "order-by-nulls-ordering.sql",
      "order-by-ordinal.sql",
      "outer-join.sql",
      "parse-schema-string.sql",
      "pivot.sql",
      "pred-pushdown.sql",
      "predicate-functions.sql",
      "query_regex_column.sql",
      "random.sql",
      "regexp-functions.sql",
      "show-create-table.sql",
      "show-tables.sql",
      "show-tblproperties.sql",
      "show-views.sql",
      "show_columns.sql",
      "sql-compatibility-functions.sql",
      "string-functions.sql",
      "struct.sql",
      "subexp-elimination.sql",
      "table-aliases.sql",
      "table-valued-functions.sql",
      "tablesample-negative.sql",
      "subquery/exists-subquery/exists-aggregate.sql",
      "subquery/exists-subquery/exists-basic.sql",
      "subquery/exists-subquery/exists-cte.sql",
      "subquery/exists-subquery/exists-having.sql",
      "subquery/exists-subquery/exists-joins-and-set-ops.sql",
      "subquery/exists-subquery/exists-orderby-limit.sql",
      "subquery/exists-subquery/exists-within-and-or.sql",
      "subquery/in-subquery/in-basic.sql",
      "subquery/in-subquery/in-group-by.sql",
      "subquery/in-subquery/in-having.sql",
      "subquery/in-subquery/in-joins.sql",
      "subquery/in-subquery/in-limit.sql",
      "subquery/in-subquery/in-multiple-columns.sql",
      "subquery/in-subquery/in-order-by.sql",
      "subquery/in-subquery/in-set-operations.sql",
      "subquery/in-subquery/in-with-cte.sql",
      "subquery/in-subquery/nested-not-in.sql",
      "subquery/in-subquery/not-in-group-by.sql",
      "subquery/in-subquery/not-in-joins.sql",
      "subquery/in-subquery/not-in-unit-tests-multi-column.sql",
      "subquery/in-subquery/not-in-unit-tests-multi-column-literal.sql",
      "subquery/in-subquery/not-in-unit-tests-single-column.sql",
      "subquery/in-subquery/not-in-unit-tests-single-column-literal.sql",
      "subquery/in-subquery/simple-in.sql",
      "subquery/negative-cases/invalid-correlation.sql",
      "subquery/negative-cases/subq-input-typecheck.sql",
      "subquery/scalar-subquery/scalar-subquery-predicate.sql",
      "subquery/scalar-subquery/scalar-subquery-select.sql",
      "subquery/subquery-in-from.sql",
      "postgreSQL/aggregates_part1.sql",
      "postgreSQL/aggregates_part2.sql",
      "postgreSQL/aggregates_part3.sql",
      "postgreSQL/aggregates_part4.sql",
      "postgreSQL/boolean.sql",
      "postgreSQL/case.sql",
      "postgreSQL/comments.sql",
      "postgreSQL/create_view.sql",
      "postgreSQL/date.sql",
      "postgreSQL/float4.sql",
      "postgreSQL/insert.sql",
      "postgreSQL/int2.sql",
      "postgreSQL/int4.sql",
      "postgreSQL/int8.sql",
      "postgreSQL/interval.sql",
      "postgreSQL/join.sql",
      "postgreSQL/limit.sql",
      "postgreSQL/numeric.sql",
      "postgreSQL/select.sql",
      "postgreSQL/select_distinct.sql",
      "postgreSQL/select_having.sql",
      "postgreSQL/select_implicit.sql",
      "postgreSQL/strings.sql",
      "postgreSQL/text.sql",
      "postgreSQL/timestamp.sql",
      "postgreSQL/union.sql",
      "postgreSQL/window_part1.sql",
      "postgreSQL/window_part2.sql",
      "postgreSQL/window_part3.sql",
      "postgreSQL/window_part4.sql",
      "postgreSQL/with.sql",
      "datetime-special.sql",
      "timestamp-ansi.sql",
      "timestamp.sql",
      "arrayJoin.sql",
      "binaryComparison.sql",
      "booleanEquality.sql",
      "caseWhenCoercion.sql",
      "concat.sql",
      "dateTimeOperations.sql",
      "decimalPrecision.sql",
      "division.sql",
      "elt.sql",
      "ifCoercion.sql",
      "implicitTypeCasts.sql",
      "inConversion.sql",
      "mapZipWith.sql",
      "mapconcat.sql",
      "promoteStrings.sql",
      "stringCastAndExpressions.sql",
      "widenSetOperationTypes.sql",
      "windowFrameCoercion.sql",
      "timestamp-ltz.sql",
      "timestamp-ntz.sql",
      "timezone.sql",
      "transform.sql",
      "try_arithmetic.sql",
      "try_cast.sql",
      "udaf.sql",
      "union.sql",
      "using-join.sql",
      // result match, but the order is not right
      // "window.sql",
      "udf/udf-union.sql",
      "udf/udf-window.sql"
    )

  private val OVERWRITE_SQL_QUERY_LIST_SPARK32: Set[String] = Set(
    // Velox corr has better computation logic but it fails Spark's precision check.
    // Remove -- SPARK-24369 multiple distinct aggregations having the same argument set
    "group-by.sql",
    // Remove -- SPARK-24369 multiple distinct aggregations having the same argument set
    "udf/udf-group-by.sql"
  )

  private val SUPPORTED_SQL_QUERY_LIST_SPARK33: Set[String] = Set(
    "bitwise.sql",
    "cast.sql",
    "change-column.sql",
    "charvarchar.sql",
    "columnresolution-negative.sql",
    "columnresolution-views.sql",
    "columnresolution.sql",
    "comments.sql",
    "comparator.sql",
    "count.sql",
    "cross-join.sql",
    "csv-functions.sql",
    "cte-legacy.sql",
    "cte-nested.sql",
    "cte-nonlegacy.sql",
    "cte.sql",
    "current_database_catalog.sql",
    "date.sql",
    "datetime-formatting-invalid.sql",
    // Velox had different handling for some illegal cases.
    //     "datetime-formatting-legacy.sql",
    //     "datetime-formatting.sql",
    "datetime-legacy.sql",
    "datetime-parsing-invalid.sql",
    "datetime-parsing-legacy.sql",
    "datetime-parsing.sql",
    "datetime-special.sql",
    "decimalArithmeticOperations.sql",
    "describe-part-after-analyze.sql",
    "describe-query.sql",
    "describe-table-after-alter-table.sql",
    "describe-table-column.sql",
    "describe.sql",
    "except-all.sql",
    "except.sql",
    "extract.sql",
    "group-by-filter.sql",
    "group-by-ordinal.sql",
    "grouping_set.sql",
    "having.sql",
    "ignored.sql",
    "inline-table.sql",
    "inner-join.sql",
    "intersect-all.sql",
    "interval.sql",
    "join-empty-relation.sql",
    "join-lateral.sql",
    "json-functions.sql",
    "like-all.sql",
    "like-any.sql",
    "limit.sql",
    "literals.sql",
    "map.sql",
    "misc-functions.sql",
    "natural-join.sql",
    "null-handling.sql",
    "null-propagation.sql",
    "operators.sql",
    "order-by-nulls-ordering.sql",
    "order-by-ordinal.sql",
    "outer-join.sql",
    "parse-schema-string.sql",
    "pivot.sql",
    "pred-pushdown.sql",
    "predicate-functions.sql",
    "query_regex_column.sql",
    "random.sql",
    "regexp-functions.sql",
    "show-create-table.sql",
    "show-tables.sql",
    "show-tblproperties.sql",
    "show-views.sql",
    "show_columns.sql",
    "sql-compatibility-functions.sql",
    "string-functions.sql",
    "struct.sql",
    "subexp-elimination.sql",
    "table-aliases.sql",
    "table-valued-functions.sql",
    "tablesample-negative.sql",
    "subquery/exists-subquery/exists-aggregate.sql",
    "subquery/exists-subquery/exists-basic.sql",
    "subquery/exists-subquery/exists-cte.sql",
    "subquery/exists-subquery/exists-having.sql",
    "subquery/exists-subquery/exists-joins-and-set-ops.sql",
    "subquery/exists-subquery/exists-orderby-limit.sql",
    "subquery/exists-subquery/exists-within-and-or.sql",
    "subquery/in-subquery/in-basic.sql",
    "subquery/in-subquery/in-group-by.sql",
    "subquery/in-subquery/in-having.sql",
    "subquery/in-subquery/in-joins.sql",
    "subquery/in-subquery/in-limit.sql",
    "subquery/in-subquery/in-multiple-columns.sql",
    "subquery/in-subquery/in-order-by.sql",
    "subquery/in-subquery/in-set-operations.sql",
    "subquery/in-subquery/in-with-cte.sql",
    "subquery/in-subquery/nested-not-in.sql",
    "subquery/in-subquery/not-in-group-by.sql",
    "subquery/in-subquery/not-in-joins.sql",
    "subquery/in-subquery/not-in-unit-tests-multi-column.sql",
    "subquery/in-subquery/not-in-unit-tests-multi-column-literal.sql",
    "subquery/in-subquery/not-in-unit-tests-single-column.sql",
    "subquery/in-subquery/not-in-unit-tests-single-column-literal.sql",
    "subquery/in-subquery/simple-in.sql",
    "subquery/negative-cases/invalid-correlation.sql",
    "subquery/negative-cases/subq-input-typecheck.sql",
    "subquery/scalar-subquery/scalar-subquery-predicate.sql",
    "subquery/scalar-subquery/scalar-subquery-select.sql",
    "subquery/subquery-in-from.sql",
    "postgreSQL/aggregates_part1.sql",
    "postgreSQL/aggregates_part2.sql",
    "postgreSQL/aggregates_part3.sql",
    "postgreSQL/aggregates_part4.sql",
    "postgreSQL/boolean.sql",
    "postgreSQL/case.sql",
    "postgreSQL/comments.sql",
    "postgreSQL/create_view.sql",
    "postgreSQL/date.sql",
    "postgreSQL/float4.sql",
    "postgreSQL/insert.sql",
    "postgreSQL/int2.sql",
    "postgreSQL/int4.sql",
    "postgreSQL/int8.sql",
    "postgreSQL/interval.sql",
    "postgreSQL/join.sql",
    "postgreSQL/limit.sql",
    "postgreSQL/numeric.sql",
    "postgreSQL/select.sql",
    "postgreSQL/select_distinct.sql",
    "postgreSQL/select_having.sql",
    "postgreSQL/select_implicit.sql",
    "postgreSQL/strings.sql",
    "postgreSQL/text.sql",
    "postgreSQL/timestamp.sql",
    "postgreSQL/union.sql",
    "postgreSQL/window_part1.sql",
    "postgreSQL/window_part2.sql",
    "postgreSQL/window_part3.sql",
    "postgreSQL/window_part4.sql",
    "postgreSQL/with.sql",
    "datetime-special.sql",
    "timestamp-ansi.sql",
    "timestamp.sql",
    "arrayJoin.sql",
    "binaryComparison.sql",
    "booleanEquality.sql",
    "caseWhenCoercion.sql",
    "concat.sql",
    "dateTimeOperations.sql",
    "decimalPrecision.sql",
    "division.sql",
    "elt.sql",
    "ifCoercion.sql",
    "implicitTypeCasts.sql",
    "inConversion.sql",
    "mapZipWith.sql",
    "mapconcat.sql",
    "promoteStrings.sql",
    "stringCastAndExpressions.sql",
    "widenSetOperationTypes.sql",
    "windowFrameCoercion.sql",
    "timestamp-ltz.sql",
    "timestamp-ntz.sql",
    "timezone.sql",
    "transform.sql",
    "try_arithmetic.sql",
    "try_cast.sql",
    "udaf.sql",
    "union.sql",
    "using-join.sql",
    "window.sql",
    "udf-union.sql",
    "udf-window.sql"
  )

  private val OVERWRITE_SQL_QUERY_LIST_SPARK33: Set[String] = Set(
    // Velox corr has better computation logic but it fails Spark's precision check.
    // Remove -- SPARK-24369 multiple distinct aggregations having the same argument set
    "group-by.sql",
    // Remove -- SPARK-24369 multiple distinct aggregations having the same argument set
    "udf/udf-group-by.sql"
  )
}
