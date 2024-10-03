CREATE TABLE IF NOT EXISTS hive.ml_tmp_db.wgame_purhis (
    open_id varchar,
    user_id varchar,
    last5bundle varchar,
    dt varchar
)  WITH (
    FORMAT = 'PARQUET',
    partitioned_by = ARRAY['dt']
);


INSERT INTO hive.ml_tmp_db.wgame_purhis

WITH purchase_history AS (
    SELECT
        *
        , TRY_CAST(JSON_PARSE(item_contents) AS ARRAY(ROW(item_goods_id VARCHAR, item_goods_count DOUBLE))) AS contents
        , ROW_NUMBER() OVER(PARTITION BY acc_id, role_id ORDER BY purchase_time DESC) AS row_num
    FROM
        kudu."10048_stage".order_attr_all_custom_kudu
    WHERE
        YEAR(TRY_CAST(purchase_time AS TIMESTAMP)) >= 2024
        AND DATE(TRY_CAST(purchase_time AS TIMESTAMP)) BETWEEN DATE_ADD('day', -30, '{{ ds }}') AND DATE_ADD('day', -1, '{{ ds }}')
        AND item_category LIKE '%礼包%'
        AND item_contents != ''
        AND svr_id LIKE 'global%'
)

, purchase_sequence_raw AS (
    SELECT
        acc_id
        , role_id
        , row_num AS idx
        , CONCAT_WS(
            U&'\0004',
            ARRAY[
                FORMAT('catid:%s', CRC32(TO_UTF8(ARBITRARY(item_category)))),
                FORMAT('price:%d', MAX(price_level))
            ] || ARRAY_AGG(
                FORMAT('type:%s,bkt:%d', c.item_type_en, CAST(TRUNCATE(LOG(2, item.item_goods_count * c.usd_price + 1)) AS INTEGER))
            )
        ) items
    FROM (
        SELECT *
        FROM purchase_history
        WHERE row_num <= 5
    ) AS his
    CROSS JOIN UNNEST(his.contents) AS item(item_goods_id, item_goods_count)
    INNER JOIN hive."ds_bi_dw".dim_wgame_bundle_item_value_config AS c 
    ON item.item_goods_id = c.item_id
    WHERE
        item.item_goods_count IS NOT NULL
        AND item.item_goods_id <> ''
    GROUP BY acc_id, role_id, row_num
)


    SELECT 
        acc_id
        , role_id
        , CONCAT_WS(
            U&'\0002'
            , ARRAY_AGG(
                CONCAT( FORMAT('last%dbundle', idx), U&'\0003', items)
            )
        ) last5bundle,
        '{{ds}}' as dt
    FROM purchase_sequence_raw
    GROUP BY acc_id, role_id