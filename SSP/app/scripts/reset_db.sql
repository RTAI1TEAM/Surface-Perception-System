DROP DATABASE IF EXISTS ML;
CREATE DATABASE ML;
USE ML;

CREATE TABLE routes (
route_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
route_name VARCHAR(100) NOT NULL,
description VARCHAR(255) NULL,
is_active TINYINT(1) NOT NULL DEFAULT 1,
loop_enabled TINYINT(1) NOT NULL DEFAULT 1,
created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
PRIMARY KEY (route_id),
UNIQUE KEY uq_routes_name (route_name)
) ENGINE=InnoDB DEFAULT CHARSET=UTF8MB4;

CREATE TABLE route_points (
    point_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    route_id BIGINT UNSIGNED NOT NULL,
    sequence_no INT NOT NULL,
    pos_x DECIMAL(10,3) NOT NULL,
    pos_y DECIMAL(10,3) NOT NULL,
    area_type VARCHAR(30) NOT NULL,
    surface_type VARCHAR(50) NOT NULL,
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (point_id),
    UNIQUE KEY uq_route_points_order (route_id, sequence_no),
    KEY idx_route_points_route (route_id),
    CONSTRAINT fk_route_points_route
        FOREIGN KEY (route_id) REFERENCES routes(route_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE indoor_route_features (
    indoor_feature_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    point_id BIGINT UNSIGNED NOT NULL,

    source_dataset VARCHAR(100) NULL,
    source_row_no INT NULL,
    scenario_name VARCHAR(100) NULL,

    series_id BIGINT NULL,

    angular_velocity_X_mean DOUBLE NOT NULL,
    angular_velocity_X_std DOUBLE NOT NULL,
    angular_velocity_X_max DOUBLE NOT NULL,
    angular_velocity_X_min DOUBLE NOT NULL,
    angular_velocity_X_median DOUBLE NOT NULL,
    angular_velocity_X_calc_range DOUBLE NOT NULL,
    angular_velocity_X_skew DOUBLE NOT NULL,
    angular_velocity_X_kurtosis DOUBLE NOT NULL,

    angular_velocity_Y_mean DOUBLE NOT NULL,
    angular_velocity_Y_std DOUBLE NOT NULL,
    angular_velocity_Y_max DOUBLE NOT NULL,
    angular_velocity_Y_min DOUBLE NOT NULL,
    angular_velocity_Y_median DOUBLE NOT NULL,
    angular_velocity_Y_calc_range DOUBLE NOT NULL,
    angular_velocity_Y_skew DOUBLE NOT NULL,
    angular_velocity_Y_kurtosis DOUBLE NOT NULL,

    angular_velocity_Z_mean DOUBLE NOT NULL,
    angular_velocity_Z_std DOUBLE NOT NULL,
    angular_velocity_Z_max DOUBLE NOT NULL,
    angular_velocity_Z_min DOUBLE NOT NULL,
    angular_velocity_Z_median DOUBLE NOT NULL,
    angular_velocity_Z_calc_range DOUBLE NOT NULL,
    angular_velocity_Z_skew DOUBLE NOT NULL,
    angular_velocity_Z_kurtosis DOUBLE NOT NULL,

    linear_acceleration_X_mean DOUBLE NOT NULL,
    linear_acceleration_X_std DOUBLE NOT NULL,
    linear_acceleration_X_max DOUBLE NOT NULL,
    linear_acceleration_X_min DOUBLE NOT NULL,
    linear_acceleration_X_median DOUBLE NOT NULL,
    linear_acceleration_X_calc_range DOUBLE NOT NULL,
    linear_acceleration_X_skew DOUBLE NOT NULL,
    linear_acceleration_X_kurtosis DOUBLE NOT NULL,

    linear_acceleration_Y_mean DOUBLE NOT NULL,
    linear_acceleration_Y_std DOUBLE NOT NULL,
    linear_acceleration_Y_max DOUBLE NOT NULL,
    linear_acceleration_Y_min DOUBLE NOT NULL,
    linear_acceleration_Y_median DOUBLE NOT NULL,
    linear_acceleration_Y_calc_range DOUBLE NOT NULL,
    linear_acceleration_Y_skew DOUBLE NOT NULL,
    linear_acceleration_Y_kurtosis DOUBLE NOT NULL,

    linear_acceleration_Z_mean DOUBLE NOT NULL,
    linear_acceleration_Z_std DOUBLE NOT NULL,
    linear_acceleration_Z_max DOUBLE NOT NULL,
    linear_acceleration_Z_min DOUBLE NOT NULL,
    linear_acceleration_Z_median DOUBLE NOT NULL,
    linear_acceleration_Z_calc_range DOUBLE NOT NULL,
    linear_acceleration_Z_skew DOUBLE NOT NULL,
    linear_acceleration_Z_kurtosis DOUBLE NOT NULL,

    roll_mean DOUBLE NOT NULL,
    roll_std DOUBLE NOT NULL,
    roll_max DOUBLE NOT NULL,
    roll_min DOUBLE NOT NULL,
    roll_median DOUBLE NOT NULL,
    roll_calc_range DOUBLE NOT NULL,
    roll_skew DOUBLE NOT NULL,
    roll_kurtosis DOUBLE NOT NULL,

    pitch_mean DOUBLE NOT NULL,
    pitch_std DOUBLE NOT NULL,
    pitch_max DOUBLE NOT NULL,
    pitch_min DOUBLE NOT NULL,
    pitch_median DOUBLE NOT NULL,
    pitch_calc_range DOUBLE NOT NULL,
    pitch_skew DOUBLE NOT NULL,
    pitch_kurtosis DOUBLE NOT NULL,

    yaw_mean DOUBLE NOT NULL,
    yaw_std DOUBLE NOT NULL,
    yaw_max DOUBLE NOT NULL,
    yaw_min DOUBLE NOT NULL,
    yaw_median DOUBLE NOT NULL,
    yaw_calc_range DOUBLE NOT NULL,
    yaw_skew DOUBLE NOT NULL,
    yaw_kurtosis DOUBLE NOT NULL,

    accel_mag_mean DOUBLE NOT NULL,
    accel_mag_std DOUBLE NOT NULL,
    accel_mag_max DOUBLE NOT NULL,
    accel_mag_min DOUBLE NOT NULL,
    accel_mag_median DOUBLE NOT NULL,
    accel_mag_calc_range DOUBLE NOT NULL,
    accel_mag_skew DOUBLE NOT NULL,
    accel_mag_kurtosis DOUBLE NOT NULL,

    gyro_mag_mean DOUBLE NOT NULL,
    gyro_mag_std DOUBLE NOT NULL,
    gyro_mag_max DOUBLE NOT NULL,
    gyro_mag_min DOUBLE NOT NULL,
    gyro_mag_median DOUBLE NOT NULL,
    gyro_mag_calc_range DOUBLE NOT NULL,
    gyro_mag_skew DOUBLE NOT NULL,
    gyro_mag_kurtosis DOUBLE NOT NULL,

    accel_diff_mean DOUBLE NOT NULL,
    accel_diff_std DOUBLE NOT NULL,
    accel_diff_max DOUBLE NOT NULL,
    accel_diff_min DOUBLE NOT NULL,
    accel_diff_median DOUBLE NOT NULL,
    accel_diff_calc_range DOUBLE NOT NULL,
    accel_diff_skew DOUBLE NOT NULL,
    accel_diff_kurtosis DOUBLE NOT NULL,

    group_id BIGINT NULL,
    surface VARCHAR(100) NULL,
    surface_encoded INT NULL,

    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (indoor_feature_id),
    UNIQUE KEY uq_indoor_feature_point (point_id),
    KEY idx_indoor_surface (surface),
    KEY idx_indoor_surface_encoded (surface_encoded),

    CONSTRAINT fk_indoor_features_point
        FOREIGN KEY (point_id) REFERENCES route_points(point_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE outdoor_route_features (
    outdoor_feature_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    point_id BIGINT UNSIGNED NOT NULL,
    source_dataset VARCHAR(100) NULL,
    source_row_no INT NULL,
    scenario_name VARCHAR(100) NULL,

    acc_mag_mean DOUBLE NOT NULL,
    acc_mag_max DOUBLE NOT NULL,
    acc_mag_skew DOUBLE NOT NULL,
    acc_mag_kurt DOUBLE NOT NULL,
    ax_mean DOUBLE NOT NULL,
    ax_skew DOUBLE NOT NULL,
    ax_kurt DOUBLE NOT NULL,
    ax_zc INT NOT NULL,
    ay_mean DOUBLE NOT NULL,
    ay_zc INT NOT NULL,
    az_std DOUBLE NOT NULL,
    az_max DOUBLE NOT NULL,
    az_skew DOUBLE NOT NULL,
    az_kurt DOUBLE NOT NULL,
    az_zc INT NOT NULL,
    gx_mean DOUBLE NOT NULL,
    gx_skew DOUBLE NOT NULL,
    gx_kurt DOUBLE NOT NULL,
    gx_zc INT NOT NULL,
    gy_skew DOUBLE NOT NULL,
    gy_kurt DOUBLE NOT NULL,
    gy_zc INT NOT NULL,
    gz_skew DOUBLE NOT NULL,
    gz_kurt DOUBLE NOT NULL,
    gz_zc INT NOT NULL,
    az_norm_mean DOUBLE NOT NULL,
    az_peak_count INT NOT NULL,
    az_crest_factor DOUBLE NOT NULL,
    acc_gyro_corr_xz DOUBLE NOT NULL,
    acc_gyro_corr_yz DOUBLE NOT NULL,
    speed_mean DOUBLE NOT NULL,
    speed_std DOUBLE NOT NULL,
    speed_x_az_energy DOUBLE NOT NULL,
    speed_bucket_high TINYINT NOT NULL,

    source_label TINYINT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (outdoor_feature_id),
    UNIQUE KEY uq_outdoor_feature_point (point_id),
    KEY idx_outdoor_feature_label (source_label),
    CONSTRAINT fk_outdoor_features_point
        FOREIGN KEY (point_id) REFERENCES route_points(point_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


CREATE TABLE prediction_logs (
    prediction_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    route_id BIGINT UNSIGNED NOT NULL,
    point_id BIGINT UNSIGNED NOT NULL,
    area_type ENUM('Indoor', 'Outdoor') NOT NULL,
    feature_table_type ENUM('indoor', 'outdoor') NOT NULL,
    indoor_feature_id BIGINT UNSIGNED NULL,
    outdoor_feature_id BIGINT UNSIGNED NULL,
    sequence_no INT NOT NULL,
    pos_x DECIMAL(10,3) NOT NULL,
    pos_y DECIMAL(10,3) NOT NULL,
    surface_type VARCHAR(50) NOT NULL,
    pred_label VARCHAR(50) NOT NULL,
    pred_prob DOUBLE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    played_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    loop_no INT NOT NULL DEFAULT 1,
    PRIMARY KEY (prediction_id),
    KEY idx_prediction_logs_route_time (route_id, played_at),
    KEY idx_prediction_logs_point_time (point_id, played_at),
    CONSTRAINT fk_prediction_logs_route
        FOREIGN KEY (route_id) REFERENCES routes(route_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_prediction_logs_point
        FOREIGN KEY (point_id) REFERENCES route_points(point_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT fk_prediction_logs_indoor_feature
        FOREIGN KEY (indoor_feature_id) REFERENCES indoor_route_features(indoor_feature_id)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    CONSTRAINT fk_prediction_logs_outdoor_feature
        FOREIGN KEY (outdoor_feature_id) REFERENCES outdoor_route_features(outdoor_feature_id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
