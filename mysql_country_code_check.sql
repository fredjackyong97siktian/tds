create table if not exists sesamedb.tds_filter_country_code_check (
    id bigint unsigned not null auto_increment,
    location_id int null,
    country_code varchar(16) not null,
    country_name varchar(100) null,
    phone_prefix varchar(24) null,
    card_country varchar(16) null,
    enabled tinyint(1) not null default 1,
    metadata json null,
    created_at datetime not null default current_timestamp,
    updated_at datetime not null default current_timestamp on update current_timestamp,
    primary key (id),
    key idx_tds_filter_country_location_enabled (location_id, enabled),
    key idx_tds_filter_country_code (country_code),
    constraint chk_tds_filter_country_has_match_value
        check (phone_prefix is not null or card_country is not null)
);

update sesamedb.tds_filter_factor
set label = 'Country code check',
    enabled = 1,
    weight = 1,
    config = null
where location_id is null
  and factor_code = 'country_code_check';

insert into sesamedb.tds_filter_factor (location_id, factor_code, label, enabled, weight, config)
select null, 'country_code_check', 'Country code check', 1, 1, null
where not exists (
    select 1
    from sesamedb.tds_filter_factor
    where location_id is null
      and factor_code = 'country_code_check'
);

grant select, insert, update, delete
on sesamedb.tds_filter_country_code_check
to 'tds_user'@'159.223.86.96';

grant select, insert, update
on sesamedb.tds_filter_factor
to 'tds_user'@'159.223.86.96';

-- Required when credit_card_entry_id resolves through entrylogs and Stripe lookup needs payment ids.
grant select
on sesamedb.entrylogs
to 'tds_user'@'159.223.86.96';
