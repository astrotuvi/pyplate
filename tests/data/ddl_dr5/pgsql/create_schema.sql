CREATE schema applause_dr5;

CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER AS $$
BEGIN
NEW.timestamp_update = NOW();
RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE applause_dr5.archive (
    archive_id           integer NOT NULL,
    archive_name         character varying(80) COLLATE ucs_basic,
    institute            character varying(80) COLLATE ucs_basic,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.archive FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.plate (
    plate_id             serial,
    archive_id           integer NOT NULL,
    plate_num            character(10) COLLATE ucs_basic,
    plate_num_orig       character(10) COLLATE ucs_basic,
    wfpdb_id             character(15) COLLATE ucs_basic,
    series               character varying(80) COLLATE ucs_basic,
    plate_format         character(7) COLLATE ucs_basic,
    plate_size1          real,
    plate_size2          real,
    emulsion             character varying(80) COLLATE ucs_basic,
    filter               character varying(80) COLLATE ucs_basic,
    development          character varying(80) COLLATE ucs_basic,
    plate_quality        character varying(80) COLLATE ucs_basic,
    plate_notes          character varying(255) COLLATE ucs_basic,
    date_orig            character varying(80) COLLATE ucs_basic,
    tz_orig              character varying(40) COLLATE ucs_basic,
    numexp               smallint,
    observatory          character varying(80) COLLATE ucs_basic,
    site_name            character varying(80) COLLATE ucs_basic,
    site_longitude       double precision,
    site_latitude        double precision,
    site_elevation       real,
    telescope            character varying(80) COLLATE ucs_basic,
    ota_name             character varying(80) COLLATE ucs_basic,
    ota_diameter         real,
    ota_aperture         real,
    ota_foclen           real,
    ota_scale            real,
    instrument           character varying(80) COLLATE ucs_basic,
    method_code          smallint,
    prism                character varying(80) COLLATE ucs_basic,
    prism_angle          character varying(10) COLLATE ucs_basic,
    dispersion           real,
    grating              character varying(80) COLLATE ucs_basic,
    air_temperature      real,
    sky_calmness         character varying(10) COLLATE ucs_basic,
    sky_sharpness        character varying(10) COLLATE ucs_basic,
    sky_transparency     character varying(10) COLLATE ucs_basic,
    sky_conditions       character varying(80) COLLATE ucs_basic,
    observer             character varying(80) COLLATE ucs_basic,
    observer_notes       character varying(255) COLLATE ucs_basic,
    notes                character varying(255) COLLATE ucs_basic,
    bibcode              character varying(80) COLLATE ucs_basic,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.plate FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.exposure (
    exposure_id          serial,
    plate_id             integer NOT NULL,
    archive_id           integer NOT NULL,
    exposure_num         smallint NOT NULL,
    object_name          character varying(80) COLLATE ucs_basic,
    object_type_code     character(2) COLLATE ucs_basic,
    ra_orig              character(11) COLLATE ucs_basic,
    dec_orig             character(11) COLLATE ucs_basic,
    flag_coord           character(1) COLLATE ucs_basic,
    ra_icrs              double precision,
    dec_icrs             double precision,
    date_orig_start      character varying(80) COLLATE ucs_basic,
    date_orig_end        character varying(80) COLLATE ucs_basic,
    time_orig_start      character varying(255) COLLATE ucs_basic,
    time_orig_end        character varying(255) COLLATE ucs_basic,
    jd_mid_orig          double precision,
    flag_time            character(1) COLLATE ucs_basic,
    ut_start             timestamp without time zone,
    ut_mid               timestamp without time zone,
    ut_weighted          timestamp without time zone,
    ut_end               timestamp without time zone,
    year_start           double precision,
    year_mid             double precision,
    year_weighted        double precision,
    year_end             double precision,
    jd_start             double precision,
    jd_mid               double precision,
    jd_weighted          double precision,
    jd_end               double precision,
    hjd_mid              double precision,
    hjd_weighted         double precision,
    exptime              real,
    num_sub              smallint,
    method_code          smallint,
    focus                real,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.exposure FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.exposure_sub (
    subexposure_id       serial,
    exposure_id          integer NOT NULL,
    plate_id             integer NOT NULL,
    exposure_num         smallint NOT NULL,
    subexposure_num      smallint NOT NULL,
    date_orig_start      character varying(10) COLLATE ucs_basic,
    time_orig_start      character varying(40) COLLATE ucs_basic,
    time_orig_end        character varying(40) COLLATE ucs_basic,
    ut_start             timestamp without time zone,
    ut_mid               timestamp without time zone,
    ut_end               timestamp without time zone,
    jd_start             double precision,
    jd_mid               double precision,
    jd_end               double precision,
    exptime              real,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.exposure_sub FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.scan (
    scan_id              serial,
    plate_id             integer NOT NULL,
    archive_id           integer NOT NULL,
    filename_scan        character varying(255) COLLATE ucs_basic,
    filename_wedge       character varying(255) COLLATE ucs_basic,
    naxis1               smallint,
    naxis2               smallint,
    minval               integer,
    maxval               integer,
    scanner              character varying(80) COLLATE ucs_basic,
    scan_res1            smallint,
    scan_res2            smallint,
    pixel_size1          real,
    pixel_size2          real,
    scan_software        character varying(80) COLLATE ucs_basic,
    scan_gamma           real,
    scan_focus           character varying(80) COLLATE ucs_basic,
    wedge                character varying(80) COLLATE ucs_basic,
    scan_date            timestamp without time zone,
    scan_author          character varying(80) COLLATE ucs_basic,
    scan_notes           character varying(255) COLLATE ucs_basic,
    origin               character varying(80) COLLATE ucs_basic,
    file_datetime        timestamp without time zone,
    file_size            bigint,
    fits_checksum        character(16) COLLATE ucs_basic,
    fits_datasum         character(10) COLLATE ucs_basic,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.scan FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.preview (
    preview_id           serial,
    plate_id             integer NOT NULL,
    archive_id           integer NOT NULL,
    preview_type         smallint,
    filename             character varying(255) COLLATE ucs_basic,
    file_format          character varying(80) COLLATE ucs_basic,
    image_width          smallint,
    image_height         smallint,
    image_datetime       timestamp without time zone,
    file_datetime        timestamp without time zone,
    file_size            integer,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.preview FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.logbook (
    logbook_id           serial,
    archive_id           integer NOT NULL,
    logbook_num          character(10) COLLATE ucs_basic,
    logbook_title        character varying(80) COLLATE ucs_basic,
    logbook_type         smallint,
    logbook_notes        character varying(255) COLLATE ucs_basic,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.logbook FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.logpage (
    logpage_id           serial,
    logbook_id           integer,
    archive_id           integer NOT NULL,
    logpage_type         smallint,
    page_num             smallint,
    logpage_order        smallint,
    filename             character varying(255) COLLATE ucs_basic,
    file_format          character varying(80) COLLATE ucs_basic,
    image_width          smallint,
    image_height         smallint,
    image_datetime       timestamp without time zone,
    file_datetime        timestamp without time zone,
    file_size            integer,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.logpage FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.plate_logpage (
    plate_id             integer NOT NULL,
    logpage_id           integer NOT NULL,
    logpage_order        smallint,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.plate_logpage FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.source (
    source_id            bigint NOT NULL,
    process_id           integer NOT NULL,
    scan_id              integer NOT NULL,
    plate_id             integer NOT NULL,
    archive_id           integer NOT NULL,
    source_num           integer,
    x_source             double precision,
    y_source             double precision,
    a_source             real,
    b_source             real,
    theta_source         real,
    erra_source          real,
    errb_source          real,
    errtheta_source      real,
    elongation           real,
    x_peak               integer,
    y_peak               integer,
    flag_usepsf          smallint,
    x_image              double precision,
    y_image              double precision,
    erra_image           real,
    errb_image           real,
    errtheta_image       real,
    x_psf                double precision,
    y_psf                double precision,
    erra_psf             real,
    errb_psf             real,
    errtheta_psf         real,
    mag_auto             real,
    magerr_auto          real,
    flux_auto            real,
    fluxerr_auto         real,
    mag_iso              real,
    magerr_iso           real,
    flux_iso             real,
    fluxerr_iso          real,
    flux_max             real,
    flux_radius          real,
    isoarea              integer,
    sqrt_isoarea         real,
    background           real,
    sextractor_flags     smallint,
    dist_center          double precision,
    dist_edge            double precision,
    annular_bin          smallint,
    flag_rim             smallint,
    flag_negradius       smallint,
    flag_clean           smallint,
    model_prediction     real,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.source FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.source_calib (
    source_id            bigint NOT NULL,
    process_id           integer,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    solution_num         smallint,
    annular_bin          smallint,
    dist_center          double precision,
    dist_edge            double precision,
    sextractor_flags     smallint,
    model_prediction     real,
    ra_icrs              double precision,
    dec_icrs             double precision,
    ra_error             real,
    dec_error            real,
    gal_lon              double precision,
    gal_lat              double precision,
    ecl_lon              double precision,
    ecl_lat              double precision,
    x_sphere             double precision,
    y_sphere             double precision,
    z_sphere             double precision,
    healpix256           integer,
    healpix1024          integer,
    nn_dist              real,
    zenith_angle         real,
    airmass              real,
    natmag               real,
    natmag_error         real,
    bpmag                real,
    bpmag_error          real,
    rpmag                real,
    rpmag_error          real,
    natmag_plate         real,
    natmag_correction    real,
    natmag_residual      real,
    phot_range_flags     smallint,
    phot_calib_flags     smallint,
    color_term           real,
    cat_natmag           real,
    match_radius         real,
    gaiadr4_id           bigint,
    gaiadr4_gmag         real,
    gaiadr4_bp_rp        real,
    gaiadr4_dist         real,
    gaiadr4_neighbors    integer,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.source_calib FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.source_xmatch (
    source_id            bigint not null,
    process_id           integer,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    dist                 real,
    solution_num         smallint,
    gaiadr4_id           bigint,
    x_gaia               double precision,
    y_gaia               double precision,
    flag_xmatch          smallint,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.source_xmatch FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.solution_set (
    solutionset_id       serial,
    process_id           integer,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    num_solutions        smallint,
    num_duplicate_solutions smallint,
    pattern_ratio        real,
    mean_pixel_scale     real,
    min_pixel_scale      real,
    max_pixel_scale      real,
    mean_fov1            real,
    mean_fov2            real,
    source_density       real,
    max_separation       real,
    header_wcs           text COLLATE ucs_basic,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.solution_set FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.solution (
    solution_id          serial,
    solutionset_id       integer,
    process_id           integer,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    solution_num         smallint,
    ra_icrs              double precision,
    dec_icrs             double precision,
    fov1                 real,
    fov2                 real,
    pixel_scale          real,
    source_density       real,
    cd1_1                double precision,
    cd1_2                double precision,
    cd2_1                double precision,
    cd2_2                double precision,
    rotation_angle       real,
    plate_mirrored       smallint,
    ncp_on_plate         smallint,
    scp_on_plate         smallint,
    stc_box              character varying(100) COLLATE ucs_basic,
    stc_polygon          character varying(200) COLLATE ucs_basic,
    header_wcs           text COLLATE ucs_basic,
    header_anet          text COLLATE ucs_basic,
    header_scamp         text COLLATE ucs_basic,
    scamp_dscale         real,
    scamp_dangle         real,
    scamp_dx             real,
    scamp_dy             real,
    scamp_sigma_1        real,
    scamp_sigma_2        real,
    scamp_sigma_mean     real,
    scamp_chi2           real,
    scamp_ndeg           integer,
    scamp_distort        integer,
    scamp_iteration      integer,
    x_centroid           real,
    y_centroid           real,
    rel_x_centroid       real,
    rel_y_centroid       real,
    num_xmatch           integer,
    num_gaia_dr4         integer,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.solution FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.solution_healpix (
    solution_id          integer,
    solutionset_id       integer,
    process_id           integer,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    solution_num         smallint,
    healpix1024          integer,
    num_sources          smallint,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.solution_healpix FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.scanner_pattern (
    pattern_id           serial,
    solutionset_id       integer,
    process_id           integer,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    axis                 smallint,
    coord                real,
    shift                real,
    extrapolated         smallint,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.scanner_pattern FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.phot_calib (
    calib_id             serial,
    process_id           integer,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    solution_num         smallint,
    iteration            smallint,
    color_term           real,
    color_term_error     real,
    cterm_stdev_fit      real,
    cterm_stdev_min      real,
    cterm_range_min      real,
    cterm_range_max      real,
    cterm_iterations     smallint,
    cterm_num_stars      integer,
    num_gaia_dr4         integer,
    num_candidate_stars  integer,
    num_calib_stars      integer,
    num_bright_stars     integer,
    num_outliers         integer,
    bright_limit         real,
    faint_limit          real,
    mag_range            real,
    rmse_min             real,
    rmse_median          real,
    rmse_max             real,
    plate_mag_brightest  real,
    plate_mag_density02  real,
    plate_mag_brightcut  real,
    plate_mag_maxden     real,
    plate_mag_lim        real,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.phot_calib FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.phot_calib_curve (
    calibcurve_id        serial,
    calib_id             integer,
    process_id           integer,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    solution_num         smallint,
    iteration            smallint,
    plate_mag            real,
    natmag               real,
    timestamp_insert     timestamp with time zone not null default NOW(),
    timestamp_update     timestamp with time zone not null default NOW()
);

CREATE TRIGGER set_timestamp BEFORE UPDATE ON applause_dr5.phot_calib_curve FOR EACH ROW EXECUTE PROCEDURE trigger_set_timestamp();

CREATE TABLE applause_dr5.process (
    process_id           serial,
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    filename             character varying(255) COLLATE ucs_basic,
    num_exposures        smallint,
    plate_epoch          real,
    hostname             character varying(80) COLLATE ucs_basic,
    timestamp_start      timestamp with time zone default NOW(),
    timestamp_end        timestamp with time zone default null,
    duration             integer,
    sky                  real,
    sky_sigma            real,
    use_psf              smallint,
    threshold            real,
    num_sources          integer,
    num_psf_sources      integer,
    num_true_sources     integer,
    num_artifacts        integer,
    solved               smallint,
    num_solutions        smallint,
    pattern_ratio        real,
    num_gaia_dr4         integer,
    color_term           real,
    bright_limit         real,
    faint_limit          real,
    mag_range            real,
    num_calib            integer,
    num_iterations       smallint,
    calibrated           smallint,
    completed            smallint,
    pyplate_version      character varying(15) COLLATE ucs_basic
);

CREATE TABLE applause_dr5.process_log (
    processlog_id        serial,
    process_id           integer NOT NULL,
    timestamp_log        timestamp with time zone not null default NOW(),
    scan_id              integer,
    plate_id             integer,
    archive_id           integer,
    level                smallint,
    event                smallint,
    solution_num         smallint,
    message              text COLLATE ucs_basic
);

