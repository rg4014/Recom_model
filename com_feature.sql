CREATE TABLE IF NOT EXISTS hive.ml_stage.wgame_bundle_comfeature (
    role_id varchar,
    acc_id varchar,
    feature varchar,
    dt varchar
) WITH (
    FORMAT = 'PARQUET',
    partitioned_by = ARRAY['dt']
);

DELETE FROM hive.ml_stage.wgame_bundle_comfeature
WHERE dt = '{{ ds }}';


INSERT INTO hive.ml_stage.wgame_bundle_comfeature
select role_id,acc_id,
array_join(array_agg(concat(k,U&'\0003',v)),U&'\0002') as fea,dt 
from 
(SELECT 
    role_id
    , acc_id
    ,map_entries(TRANSFORM_VALUES(MAP_FILTER(f, (x, v) -> CONTAINS(SPLIT('{{feature_list}}', ','), x)), (k,v) -> v[1])) AS TT
    , dt
FROM (
    SELECT 
        role_id, acc_id, dt, SPLIT_TO_MULTIMAP(feature, U&'\0002', U&'\0003') AS f
    FROM hive."10048_ml_dw".bundle_price_role_common_feature_dd 
    WHERE dt = '{{ ds }}' 
        AND feature_group = 'attribute' 
))t
CROSS JOIN UNNEST(TT) as t (k,v)

group by role_id ,acc_id,dt





