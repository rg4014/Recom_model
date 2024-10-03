CREATE TABLE IF NOT EXISTS hive.ml_tmp_db.wgame_bundle_dp_resfeature (
    role_id varchar,
    acc_id varchar,
    feature VARCHAR,
    feature_group varchar,
    dt varchar
) WITH (
    FORMAT = 'PARQUET',
    partitioned_by = ARRAY['feature_group','dt']
);


DELETE FROM hive.ml_tmp_db.wgame_bundle_dp_resfeature
WHERE dt = '{{ ds }}'
AND feature_group = 'res_ratity';

INSERT INTO hive.ml_tmp_db.wgame_bundle_dp_resfeature

WITH role_feature AS(
	SELECT  
        current_acc_id                                                                                                                               AS acc_id
        ,role_id
        ,svr_region
        ,base_level
        ,vip_level
        ,research_building_level
        ,city_level
        ,chapter_task
        ,coalesce(datediff('{{ ds }}' ,born_time) ,0)                                                                                                         AS lifespan
        ,recharge
        ,gold + gold_card_5k * 5000 + oil + oil_card_5k * 5000 + metal + metal_card_5k * 5000 + chest_select_resource_5k * 5000                       AS resource
        ,resource_out_7d
        ,rush_build_common + rush_tech_common + rush_common                                                                                           AS rush_common
        ,rush_common_out_7d
        ,top_key
        ,top_key_out_7d
        ,officer_bar_ticket
        ,officer_bar_ticket_out_7d
        ,IF(common_equipment_exp + common_equipment_exp_consume_levelup_1 = 0,1,(common_equipment_exp - common_equipment_exp_consume_levelup_1) * 1e0/ (common_equipment_exp + common_equipment_exp_consume_levelup_1)) AS common_equipment_exp_consume_levelup_1_diff_num_relatively
        ,IF(common_equipment_exp + common_equipment_exp_consume_blueprint_1 = 0,1,(common_equipment_exp - common_equipment_exp_consume_blueprint_1) * 1e0/ (common_equipment_exp + common_equipment_exp_consume_blueprint_1)) AS common_equipment_exp_consume_blueprint_1_diff_num_relatively
        ,IF(ammunition + ammunition_consume_levelup_1 = 0,1,(ammunition - ammunition_consume_levelup_1) * 1e0/ (ammunition + ammunition_consume_levelup_1)) AS ammunition_consume_levelup_1_diff_num_relatively
        ,IF(ammunition + ammunition_consume_levelup_10 = 0,1,(ammunition - ammunition_consume_levelup_10) * 1e0/ (ammunition + ammunition_consume_levelup_10)) AS ammunition_consume_levelup_10_diff_num_relatively
        ,IF(ammunition + ammunition_consume_blueprint_1 = 0,1,(ammunition - ammunition_consume_blueprint_1) * 1e0/ (ammunition + ammunition_consume_blueprint_1)) AS ammunition_consume_blueprint_1_diff_num_relatively
        ,IF(ammunition + ammunition_consume_blueprint_10 = 0,1,(ammunition - ammunition_consume_blueprint_10) * 1e0/ (ammunition + ammunition_consume_blueprint_10)) AS ammunition_consume_blueprint_10_diff_num_relatively
        ,IF(elements + elements_consume_blueprint_1 = 0,1,(elements - elements_consume_blueprint_1) * 1e0/ (elements + elements_consume_blueprint_1)) AS elements_consume_blueprint_1_diff_num_relatively
        ,IF(elements + elements_consume_blueprint_10 = 0,1,(elements - elements_consume_blueprint_10) * 1e0/ (elements + elements_consume_blueprint_10)) AS elements_consume_blueprint_10_diff_num_relatively
        ,IF(elements + elements_consume_reform = 0,1,(elements - elements_consume_reform) * 1e0/ (elements + elements_consume_reform))                AS elements_consume_reform_diff_num_relatively
        ,IF(airforce_equipment_exp + airforce_equipment_exp_consume_levelup_1 = 0,1,(airforce_equipment_exp - airforce_equipment_exp_consume_levelup_1) * 1e0/ (airforce_equipment_exp + airforce_equipment_exp_consume_levelup_1)) AS airforce_equipment_exp_consume_levelup_1_diff_num_relatively
        ,IF(airforce_equipment_exp + airforce_equipment_exp_consume_blueprint_1 = 0,1,(airforce_equipment_exp - airforce_equipment_exp_consume_blueprint_1) * 1e0/ (airforce_equipment_exp + airforce_equipment_exp_consume_blueprint_1)) AS airforce_equipment_exp_consume_blueprint_1_diff_num_relatively
        ,IF(airforce_equipment_exp + airforce_equipment_exp_consume_blueprint_10 = 0,1,(airforce_equipment_exp - airforce_equipment_exp_consume_blueprint_10) * 1e0/ (airforce_equipment_exp + airforce_equipment_exp_consume_blueprint_10)) AS airforce_equipment_exp_consume_blueprint_10_diff_num_relatively
        ,IF(airforce_ammunition + airforce_ammunition_consume_levelup_1 = 0,1,(airforce_ammunition - airforce_ammunition_consume_levelup_1) * 1e0/ (airforce_ammunition + airforce_ammunition_consume_levelup_1)) AS airforce_ammunition_consume_levelup_1_diff_num_relatively
        ,IF(airforce_ammunition + airforce_ammunition_consume_levelup_to_10 = 0,1,(airforce_ammunition - airforce_ammunition_consume_levelup_to_10) * 1e0/ (airforce_ammunition + airforce_ammunition_consume_levelup_to_10)) AS airforce_ammunition_consume_levelup_to_10_diff_num_relatively
        ,IF(airforce_ammunition + airforce_ammunition_consume_levelup_10 = 0,1,(airforce_ammunition - airforce_ammunition_consume_levelup_10) * 1e0/ (airforce_ammunition + airforce_ammunition_consume_levelup_10)) AS airforce_ammunition_consume_levelup_10_diff_num_relatively
        ,IF(airforce_ammunition + airforce_ammunition_consume_blueprint_1 = 0,1,(airforce_ammunition - airforce_ammunition_consume_blueprint_1) * 1e0/ (airforce_ammunition + airforce_ammunition_consume_blueprint_1)) AS airforce_ammunition_consume_blueprint_1_diff_num_relatively
        ,IF(airforce_ammunition + airforce_ammunition_consume_blueprint_10 = 0,1,(airforce_ammunition - airforce_ammunition_consume_blueprint_10) * 1e0/ (airforce_ammunition + airforce_ammunition_consume_blueprint_10)) AS airforce_ammunition_consume_blueprint_10_diff_num_relatively
        ,IF(airforce_elements + airforce_elements_consume_blueprint_1 = 0,1,(airforce_elements - airforce_elements_consume_blueprint_1) * 1e0/ (airforce_elements + airforce_elements_consume_blueprint_1)) AS airforce_elements_consume_blueprint_1_diff_num_relatively
        ,IF(airforce_elements + airforce_elements_consume_blueprint_10 = 0,1,(airforce_elements - airforce_elements_consume_blueprint_10) * 1e0/ (airforce_elements + airforce_elements_consume_blueprint_10)) AS airforce_elements_consume_blueprint_10_diff_num_relatively
        ,IF(airforce_elements + airforce_elements_consume_reform = 0,1,(airforce_elements - airforce_elements_consume_reform) * 1e0/ (airforce_elements + airforce_elements_consume_reform)) AS airforce_elements_consume_reform_diff_num_relatively
        ,IF(replacement_wrench + replacement_wrench_consume_mechanic_1 = 0,1,(replacement_wrench - replacement_wrench_consume_mechanic_1) * 1e0/ (replacement_wrench + replacement_wrench_consume_mechanic_1)) AS replacement_wrench_consume_mechanic_1_diff_num_relatively
        ,IF(officer_exp * 10000 + officer_exp_consume_lvlup_1 = 0,1,(officer_exp * 10000 - officer_exp_consume_lvlup_1) * 1e0/ (officer_exp *10000 + officer_exp_consume_lvlup_1)) AS officer_exp_consume_lvlup_1_diff_num_relatively
        ,IF(officer_exp + officer_exp_consume_lvlup_to_10 = 0,1,(officer_exp - officer_exp_consume_lvlup_to_10) * 1e0/ (officer_exp + officer_exp_consume_lvlup_to_10)) AS officer_exp_consume_lvlup_to_10_diff_num_relatively
        ,IF((component_casting + component_stamping + component_machining + component_lightindustrial + component_chemical + chest_choice_replacement) + (component_casting_consume + component_stamping_consume + component_machining_consume + component_lightindustrial_consume + component_chemical_consume) = 0,1,((component_casting + component_stamping + component_machining + component_lightindustrial + component_chemical + chest_choice_replacement) - (component_casting_consume + component_stamping_consume + component_machining_consume + component_lightindustrial_consume + component_chemical_consume)) * 1e0/ ((component_casting + component_stamping + component_machining + component_lightindustrial + component_chemical + chest_choice_replacement) + (component_casting_consume + component_stamping_consume + component_machining_consume + component_lightindustrial_consume + component_chemical_consume))) AS component_consume_diff_num_relatively
	FROM hive."10048_ml_dw".role_iris_res_rarity_daily
	WHERE current_acc_id <> ''
        AND DATE(dt) BETWEEN DATE_ADD('day', -1, '{{ ds }}') AND DATE_ADD('day', 1, '{{ ds }}')
        AND DATE(TRY_CAST(update_time AS TIMESTAMP)) = DATE('{{ ds }}')
)

, f AS (
    SELECT  
        acc_id
        ,role_id
        ,svr_region
        ,base_level
        ,vip_level
        ,research_building_level
        ,city_level
        ,chapter_task
        ,lifespan
        ,IF(resource + resource_out_7d <= 0, 1, 0.5 - (resource - resource_out_7d/7)*1e0/(resource + resource_out_7d/7)/2 ) AS resource_rarity
        ,IF(rush_common + rush_common_out_7d <= 0, 1, 0.5 - (rush_common - rush_common_out_7d/7)*1e0/(rush_common + rush_common_out_7d/7)/2 ) AS speedup_rarity
        ,IF(top_key + top_key_out_7d <= 0, 1, 0.5 - (top_key - top_key_out_7d/7)*1e0/(top_key + top_key_out_7d/7)/2 ) AS top_key_rarity
        ,IF(officer_bar_ticket + officer_bar_ticket_out_7d <= 0, 1, 0.5 - (officer_bar_ticket - officer_bar_ticket_out_7d/7)*1e0/(officer_bar_ticket + officer_bar_ticket_out_7d/7)/2 ) AS officer_bar_ticket_rarity
        ,common_equipment_exp_consume_levelup_1_diff_num_relatively
        ,common_equipment_exp_consume_blueprint_1_diff_num_relatively
        ,(1 - LEAST(common_equipment_exp_consume_levelup_1_diff_num_relatively, common_equipment_exp_consume_blueprint_1_diff_num_relatively))/2 AS common_equipment_exp_rarity
        ,airforce_equipment_exp_consume_levelup_1_diff_num_relatively
        ,airforce_equipment_exp_consume_blueprint_1_diff_num_relatively
        ,(1 - LEAST(airforce_equipment_exp_consume_levelup_1_diff_num_relatively, airforce_equipment_exp_consume_blueprint_1_diff_num_relatively))/2 AS airforce_equipment_exp_rarity
        ,ammunition_consume_levelup_1_diff_num_relatively
        ,ammunition_consume_levelup_10_diff_num_relatively
        ,ammunition_consume_blueprint_1_diff_num_relatively
        ,ammunition_consume_blueprint_10_diff_num_relatively
        ,(1 - LEAST(ammunition_consume_levelup_1_diff_num_relatively, ammunition_consume_blueprint_1_diff_num_relatively))/2 AS ammunition_rarity
        ,airforce_ammunition_consume_levelup_1_diff_num_relatively
        ,airforce_ammunition_consume_levelup_10_diff_num_relatively
        ,airforce_ammunition_consume_blueprint_1_diff_num_relatively
        ,airforce_ammunition_consume_blueprint_10_diff_num_relatively
        ,(1 - LEAST(airforce_ammunition_consume_levelup_1_diff_num_relatively, airforce_ammunition_consume_blueprint_1_diff_num_relatively))/2 AS airforce_ammunition_rarity
        ,elements_consume_blueprint_1_diff_num_relatively
        ,elements_consume_blueprint_10_diff_num_relatively
        ,elements_consume_reform_diff_num_relatively
        ,(1 - LEAST(elements_consume_blueprint_1_diff_num_relatively, elements_consume_reform_diff_num_relatively))/2 AS elements_rarity
        ,airforce_elements_consume_blueprint_1_diff_num_relatively
        ,airforce_elements_consume_blueprint_10_diff_num_relatively
        ,airforce_elements_consume_reform_diff_num_relatively
        ,(1 - LEAST(airforce_elements_consume_blueprint_1_diff_num_relatively, airforce_elements_consume_reform_diff_num_relatively))/2 AS airforce_elements_rarity
        ,replacement_wrench_consume_mechanic_1_diff_num_relatively
        ,(1 - replacement_wrench_consume_mechanic_1_diff_num_relatively)/2 AS replacement_wrench_rarity
        ,officer_exp_consume_lvlup_1_diff_num_relatively
        ,(1 - officer_exp_consume_lvlup_1_diff_num_relatively)/2 AS officer_exp_rarity
        ,component_consume_diff_num_relatively
        ,(1 - component_consume_diff_num_relatively)/2 AS chest_choice_replacement_rarity
    FROM role_feature
)

SELECT  
    role_id
    , acc_id
    ,CONCAT_WS(U&'\0002',
           CONCAT('svr_region'\ svr_region),
           CONCAT('base_level', U&'\0003', FORMAT_NUMBER(base_level)),
           CONCAT('vip_level', U&'\0003', FORMAT_NUMBER(vip_level)),
           CONCAT('research_building_level', U&'\0003', FORMAT_NUMBER(research_building_level)),
           CONCAT('city_level', U&'\0003', city_level),
           CONCAT('chapter_task', U&'\0003', FORMAT_NUMBER(chapter_task)),
           CONCAT('lifespan', U&'\0003', FORMAT_NUMBER(lifespan)),
           CONCAT('resource_rarity', U&'\0003', FORMAT('%.2f' ,resource_rarity)),
           CONCAT('speedup_rarity', U&'\0003', FORMAT('%.2f' ,speedup_rarity)),
           CONCAT('top_key_rarity', U&'\0003', FORMAT('%.2f' ,top_key_rarity)),
           CONCAT('officer_bar_ticket_rarity', U&'\0003', FORMAT('%.2f' ,officer_bar_ticket_rarity)),
           CONCAT('common_equipment_exp_consume_levelup_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,common_equipment_exp_consume_levelup_1_diff_num_relatively)),
           CONCAT('common_equipment_exp_consume_blueprint_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,common_equipment_exp_consume_blueprint_1_diff_num_relatively)),
           CONCAT('common_equipment_exp_rarity', U&'\0003', FORMAT('%.2f' ,common_equipment_exp_rarity)),
           CONCAT('airforce_equipment_exp_consume_levelup_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_equipment_exp_consume_levelup_1_diff_num_relatively)),
           CONCAT('airforce_equipment_exp_consume_blueprint_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_equipment_exp_consume_blueprint_1_diff_num_relatively)),
           CONCAT('airforce_equipment_exp_rarity', U&'\0003', FORMAT('%.2f' ,airforce_equipment_exp_rarity)),
           CONCAT('ammunition_consume_levelup_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,ammunition_consume_levelup_1_diff_num_relatively)),
           CONCAT('ammunition_consume_levelup_10_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,ammunition_consume_levelup_10_diff_num_relatively)),
           CONCAT('ammunition_consume_blueprint_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,ammunition_consume_blueprint_1_diff_num_relatively)),
           CONCAT('ammunition_consume_blueprint_10_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,ammunition_consume_blueprint_10_diff_num_relatively)),
           CONCAT('ammunition_rarity', U&'\0003', FORMAT('%.2f' ,ammunition_rarity)),
           CONCAT('airforce_ammunition_consume_levelup_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_ammunition_consume_levelup_1_diff_num_relatively)),
           CONCAT('airforce_ammunition_consume_levelup_10_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_ammunition_consume_levelup_10_diff_num_relatively)),
           CONCAT('airforce_ammunition_consume_blueprint_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_ammunition_consume_blueprint_1_diff_num_relatively)),
           CONCAT('airforce_ammunition_consume_blueprint_10_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_ammunition_consume_blueprint_10_diff_num_relatively)),
           CONCAT('airforce_ammunition_rarity', U&'\0003', FORMAT('%.2f' ,airforce_ammunition_rarity)),
           CONCAT('elements_consume_blueprint_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,elements_consume_blueprint_1_diff_num_relatively)),
           CONCAT('elements_consume_blueprint_10_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,elements_consume_blueprint_10_diff_num_relatively)),
           CONCAT('elements_consume_reform_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,elements_consume_reform_diff_num_relatively)),
           CONCAT('elements_rarity', U&'\0003', FORMAT('%.2f' ,elements_rarity)),
           CONCAT('airforce_elements_consume_blueprint_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_elements_consume_blueprint_1_diff_num_relatively)),
           CONCAT('airforce_elements_consume_blueprint_10_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_elements_consume_blueprint_10_diff_num_relatively)),
           CONCAT('airforce_elements_consume_reform_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,airforce_elements_consume_reform_diff_num_relatively)),
           CONCAT('airforce_elements_rarity', U&'\0003', FORMAT('%.2f' ,airforce_elements_rarity)),
           CONCAT('replacement_wrench_consume_mechanic_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,replacement_wrench_consume_mechanic_1_diff_num_relatively)),
           CONCAT('replacement_wrench_rarity', U&'\0003', FORMAT('%.2f' ,replacement_wrench_rarity)),
           CONCAT('officer_exp_consume_lvlup_1_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,officer_exp_consume_lvlup_1_diff_num_relatively)),
           CONCAT('officer_exp_rarity', U&'\0003', FORMAT('%.2f' ,officer_exp_rarity)),
           CONCAT('component_consume_diff_num_relatively', U&'\0003', FORMAT('%.2f' ,component_consume_diff_num_relatively)),
           CONCAT('chest_choice_replacement_rarity', U&'\0003', FORMAT('%.2f' ,chest_choice_replacement_rarity))
        ) AS feature

    , 'res_ratity'
    , '{{ ds }}'
FROM f

