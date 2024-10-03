-- 增加了一些item remain的相关特征
CREATE TABLE IF NOT EXISTS hive.ml_tmp_db.wgame_bundle_price(
    open_id varchar,
    user_id varchar,
    actions varchar,
    serverid varchar,
    recharge double,
    bundle_price varchar,
    bundle_type varchar,
    his30maxpay integer,  
    his30totalpay integer,
    his30minpay integer,
    event_time varchar,
    intervene_type varchar,
    gift_contents varchar,
   dt varchar
)  WITH (
    FORMAT = 'PARQUET',
    partitioned_by = ARRAY['dt']
);

DELETE FROM hive.ml_tmp_db.wgame_bundle_price
WHERE dt = '{{ ds }}';

INSERT INTO hive.ml_tmp_db.wgame_bundle_price
SELECT 
		pay.open_id,
        pay.user_id,
        fea.action,
        fea.serverid,
        fea.recharge,
        fea.bundle_price,
        fea.trigger_type,
        pay.max_bundle_price_condition,
        pay.sum_bundle_price_condition,
        pay.min_bundle_price_condition,
        fea.event_time,
        fea.intervene_type,
        fea.gift_contents,
        pay.dt
        

FROM
(SELECT
        open_id,
        user_id,
        MAX(
        TRY_CAST(bundle_price AS integer) * (CASE WHEN action = 'spc_bundle_buy' THEN 1 ELSE 0 END)
    ) AS max_bundle_price_condition,
    SUM(
        TRY_CAST(bundle_price AS integer) * (CASE WHEN action = 'spc_bundle_buy' THEN 1 ELSE 0 END)
    ) AS sum_bundle_price_condition,
    MIN(CASE WHEN action = 'spc_bundle_buy' THEN TRY_CAST(bundle_price as integer) END) AS min_bundle_price_condition,
        '{{ ds }}' AS dt
    FROM
        hive."10048_oss_bi_dw".wgame_trigger_bundle_v2
    WHERE
        DATE(dt) BETWEEN DATE_ADD('day', -30, '{{ ds }}') AND DATE_ADD('day', -1, '{{ ds }}')
        AND os_type != ''

    GROUP BY open_id, user_id) as pay

INNER JOIN
    (SELECT
        open_id,
        user_id,
        action,
        event_time,
        SPLIT_PART(server_id, '-', 1) as serverid,
        
        recharge,
        bundle_price,
        trigger_type,
        intervene_type,
        gift_contents,
        map_entries(TRANSFORM_VALUES(MAP_FILTER(f, (x, v) -> CONTAINS(SPLIT('{{feature_list}}', ','), x)), (k,v) -> v[1])) AS TT
        '{{ ds }}' AS dt
        
    FROM
        hive."10048_oss_bi_dw".wgame_trigger_bundle_v2
    WHERE
     		dt='{{ds}}' AND
        os_type != ''
    ) as fea
    on pay.open_id=fea.open_id and pay.user_id=fea.user_id and pay.dt=fea.dt



