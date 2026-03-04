import ee

def counties_fc(state_fips: str, county_fips_filter: str | None):
    # EE must be initialized before calling this
    state_fips = str(state_fips).zfill(2)

    fc = ee.FeatureCollection("TIGER/2018/Counties").filter(
        ee.Filter.eq("STATEFP", state_fips)
    )

    if county_fips_filter:
        # county must be 3 digits
        county_fips = str(county_fips_filter).zfill(3)
        geoid = state_fips + county_fips
        fc = fc.filter(ee.Filter.eq("GEOID", geoid))

    return fc
