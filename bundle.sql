CREATE TABLE IF NOT EXISTS hive.ml_stage.wgame_bundle_content (
    open_id varchar,
    user_id varchar,
    event_time varchar,
    content varchar,
    intervene_type varchar,
    dt varchar
) WITH (
    FORMAT = 'PARQUET',
    partitioned_by = ARRAY['dt']
);

INSERT INTO hive.ml_stage.wgame_bundle_content
SELECT 
    open_id, user_id,event_time,intervene_type,
    CONCAT_WS(U&'\0002',
        ARRAY_AGG(
            CONCAT(
                item_goods_id, 
                U&'\0003', 
                CAST(item_goods_count AS VARCHAR)
            )
        )
       
    ) AS bundle_content,
    '{{ds}}' AS dt
FROM 
(
    SELECT 
        event_time, open_id, user_id,intervene_type,
        item_goods_id,
        item_goods_count
    FROM 
    (
        SELECT * 
        FROM hive."10048_oss_bi_dw".wgame_trigger_bundle_v2
        WHERE dt='{{ds}}'
    ) t
    CROSS JOIN UNNEST(t.gift_contents)
)
GROUP BY event_time, open_id, user_id,intervene_type 