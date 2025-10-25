import math
from datetime import datetime
import json
import time 

# ----------------------------------------------------------------------
# 📜 IMPLEMENTATION FOR: Citizens Property Insurance Corporation (public.pdf)
# ----------------------------------------------------------------------

def calculate_citizens_dp3_premium(policy_details: dict) -> dict:
    """
    Calculates the DP-3 premium based on the Citizens (public.pdf) manual.
    This is the public, last-resort insurer.
    """
    prop = policy_details.get("property", {})
    policyholder = policy_details.get("policyholder", {})
    county = prop.get("county")
    if not county: return {"error": "County is required."}

    # --- Eligibility Check ---
    # Citizens has more lenient roof age rules than private insurers.
    # We will assume a general 30-year limit for this model.
    # Current year is 2025 based on document dates.
    roof_age = 2025 - prop.get("roof_year", 2000)
    if roof_age > 30:
        return {
            "final_premium": 0,
            "insurer": "Citizens Property Insurance",
            "note": f"Ineligible: Roof age ({roof_age} years) exceeds the 30-year limit for this model."
        }
    
    # Helper functions encapsulated for clarity
    def get_territory_info(c):
        territory_map = {
            'Alachua': 2, 'Baker': 2, 'Bay': 7, 'Bradford': 2, 'Brevard': 14, 'Broward': 23, 'Calhoun': 7,
            'Charlotte': 20, 'Citrus': 10, 'Clay': 3, 'Collier': 22, 'Columbia': 2, 'DeSoto': 11,
            'Dixie': 8, 'Duval': 4, 'Escambia': 6, 'Flagler': 5, 'Franklin': 7, 'Gadsden': 1,
            'Gilchrist': 2, 'Glades': 12, 'Gulf': 7, 'Hamilton': 2, 'Hardee': 11, 'Hendry': 12,
            'Hernando': 10, 'Highlands': 11, 'Hillsborough': 17, 'Holmes': 7, 'Indian River': 15,
            'Jackson': 7, 'Jefferson': 1, 'Lafayette': 2, 'Lake': 13, 'Lee': 21, 'Leon': 1, 'Levy': 8,
            'Liberty': 7, 'Madison': 2, 'Manatee': 18, 'Marion': 9, 'Martin': 16, 'Miami-Dade': 24,
            'Monroe': 25, 'Nassau': 4, 'Okaloosa': 6, 'Okeechobee': 12, 'Orange': 13, 'Osceola': 13,
            'Palm Beach': 23, 'Pasco': 17, 'Pinellas': 19, 'Polk': 11, 'Putnam': 3, 'Santa Rosa': 6,
            'Sarasota': 18, 'Seminole': 13, 'St. Johns': 5, 'St. Lucie': 16, 'Sumter': 9,
            'Suwannee': 2, 'Taylor': 8, 'Union': 2, 'Volusia': 14, 'Wakulla': 7, 'Walton': 6, 'Washington': 7
        }
        return {"territory_id": territory_map.get(c.title(), 2)}

    def get_base_rate(tid, cov_a):
        rate_table_sample = {
            2: [(200000, 1636), (300000, 2307), (400000, 2977)],
            17: [(200000, 4771), (300000, 6808), (400000, 8845)],
            24: [(200000, 9152), (300000, 13327), (400000, 17502)],
        }
        rates = rate_table_sample.get(tid, rate_table_sample[2])
        if cov_a < rates[0][0]: return rates[0][1] * (cov_a / rates[0][0])
        for i in range(len(rates) - 1):
            c1, r1 = rates[i]; c2, r2 = rates[i+1]
            if c1 <= cov_a <= c2: return r1 + ((cov_a - c1) / (c2 - c1)) * (r2 - r1)
        c1, r1 = rates[-2]; c2, r2 = rates[-1]
        return r2 + (cov_a - c2) * ((r2 - r1) / (c2 - c1))

    # Calculation logic
    territory_id = get_territory_info(county)["territory_id"]
    coverage_a = prop.get("coverage_a_amount", 250000)
    base_premium = get_base_rate(territory_id, coverage_a)

    def get_construction_factor(c, y):
        ctype = c.lower(); return 0.53 if "frame" in ctype and y >= 2002 else (0.44 if "masonry" in ctype and y >= 2002 else (0.88 if "masonry" in ctype else 1.0))
    def get_deductible_factor(d): return {500: 1.10, 1000: 1.00, 2500: 0.82, 5000: 0.70}.get(d, 1.0)
    def get_mitigation_factor(r): return {'A': 0.69, 'B': 0.72, 'C': 0.77}.get(r.upper(), 1.0)

    subtotal_a = base_premium * get_construction_factor(prop.get("construction_type", "Frame"), prop.get("year_built", 1990))
    subtotal_a *= get_deductible_factor(policyholder.get("deductible", 1000))
    subtotal_a *= get_mitigation_factor(prop.get("wind_mitigation_rating", "None"))
    
    subtotal_b = round(subtotal_a) + 50 # Add sinkhole charge
    final_premium = round(subtotal_b * 1.01 + 22) # Add hurricane surcharge and policy fee
    
    return {"final_premium": final_premium, "insurer": "Citizens Property Insurance"}

# ----------------------------------------------------------------------
# 📜 IMPLEMENTATION FOR: Universal Property (private2.pdf)
# ----------------------------------------------------------------------

def calculate_universal_dp3_premium(policy_details: dict) -> dict:
    """Calculates the DP-3 premium based on the Universal Property (private2.pdf) manual."""
    prop = policy_details.get("property", {})
    policyholder = policy_details.get("policyholder", {})
    county = prop.get("county")
    if not county: return {"error": "County is required."}

    # --- Eligibility Check ---
    # Private insurers are very strict about roof age. Assume a 20-year limit for DP-3.
    # Current year is 2025 based on document dates.
    roof_age = 2025 - prop.get("roof_year", 2000)
    if roof_age > 20:
        return {
            "final_premium": 0,
            "insurer": "Universal Property",
            "note": f"Ineligible: Roof age ({roof_age} years) exceeds the 20-year limit for this model."
        }

    def get_territory_info(c): return {"rate": {'Miami-Dade': 6927, 'Broward': 5845, 'Hillsborough': 3317, 'Alachua': 1205, 'Pinellas': 3317, 'Orange': 2950}.get(c.title(), 1205)}
    
    base_premium = get_territory_info(county)["rate"]
    final_premium = base_premium
    
    def get_age_factor(y):
        # Current year is 2025 based on document dates.
        age = 2025 - y
        if age <= 5: return 0.75;
        if age <= 10: return 0.85;
        if age <= 20: return 1.00;
        return 1.20
    def get_score_factor(s): return {1: 0.78, 2: 0.81, 3: 0.85, 4: 0.90, 5: 0.95, 6: 1.00, 7: 1.05, 8: 1.10, 9: 1.15, 10: 1.20}.get(s, 1.0)
    
    # Assumption: Base Rate is for a $100,000 baseline home.
    final_premium *= (prop.get("coverage_a_amount", 250000) / 100000.0)
    final_premium *= 0.90 if "masonry" in prop.get("construction_type", "").lower() else 1.0
    final_premium *= get_age_factor(prop.get("year_built", 2000))
    final_premium *= get_score_factor(policyholder.get("insurance_score_tier", 6))

    return {"final_premium": round(final_premium), "insurer": "Universal Property"}

# ----------------------------------------------------------------------
# 📜 IMPLEMENTATION FOR: State Farm (private.pdf)
# ----------------------------------------------------------------------
def calculate_statefarm_dp3_premium(policy_details: dict) -> dict:
    """
    Calculates a DP-3 premium based on private.pdf.
    [cite_start]NOTE: This document is a 'Billing Payment Agreement' [cite: 506] [cite_start]for the 'Rental Dwelling Program'[cite: 561, 664], not a full rate manual.
    The logic below is a simplified model created to be functional, as no base rates are provided.
    [cite_start]The filing does mention adding a late fee of $10 and a returned payment fee of $15[cite: 516].
    """
    prop = policy_details.get("property", {})
    policyholder = policy_details.get("policyholder", {})
    county = prop.get("county")
    if not county: return {"error": "County is required."}

    # --- Eligibility Check ---
    # Assume a standard 20-year roof age limit for a private insurer.
    # Current year is 2025 based on document dates.
    roof_age = 2025 - prop.get("roof_year", 2000)
    if roof_age > 20:
        return {
            "final_premium": 0,
            "insurer": "State Farm",
            "note": f"Ineligible: Roof age ({roof_age} years) exceeds the 20-year limit for this model."
        }

    # ASSUMPTION: Create a simplified base rate structure as none is provided in the document.
    # These are illustrative values.
    def get_county_base_rate(c):
        rate_map = {'Hillsborough': 3100, 'Pinellas': 3250, 'Alachua': 1150, 'Orange': 2800}
        return rate_map.get(c.title(), 1200)

    # Calculation based on simplified model
    base_rate = get_county_base_rate(county)
    coverage_a = prop.get("coverage_a_amount", 250000)
    
    # Model premium as proportional to Coverage A, using $200,000 as a baseline.
    final_premium = base_rate * (coverage_a / 200000.0)

    # The document's primary purpose is to add fees. While not part of the upfront premium,
    # they are a key part of the filing. We'll note them.
    note = "Premium is an estimate. Filing adds a $10 late fee and a $15 returned payment fee. [cite: 516, 533, 542]"

    return {"final_premium": round(final_premium), "insurer": "State Farm", "note": note}

# ----------------------------------------------------------------------
# 📜 IMPLEMENTATION FOR: Progressive / ASI (private3.pdf)
# ----------------------------------------------------------------------
def calculate_progressive_dp3_premium(policy_details: dict) -> dict:
    """
    Calculates a DP-3 premium based on private3.pdf.
    [cite_start]NOTE: This document is a 'Dwelling Notices Update' filing[cite: 9], not a rate manual.
    The logic is a simplified model based on components mentioned in the notice templates.
    """
    prop = policy_details.get("property", {})
    policyholder = policy_details.get("policyholder", {})
    county = prop.get("county")
    if not county: return {"error": "County is required."}

    # --- Eligibility Check ---
    # Assume a standard 20-year roof age limit for a private insurer.
    # Current year is 2025 based on document dates.
    roof_age = 2025 - prop.get("roof_year", 2000)
    if roof_age > 20:
        return {
            "final_premium": 0,
            "insurer": "Progressive / ASI",
            "note": f"Ineligible: Roof age ({roof_age} years) exceeds the 20-year limit for this model."
        }

    # ASSUMPTION: Create a simplified base premium since none is provided.
    # We will model it as a function of Coverage A and county.
    def get_base_premium(cov_a, c):
        county_rate_factor = {'Hillsborough': 0.015, 'Pinellas': 0.016, 'Alachua': 0.008, 'Orange': 0.012}
        factor = county_rate_factor.get(c.title(), 0.009)
        return cov_a * factor

    coverage_a = prop.get("coverage_a_amount", 250000)
    base_premium = get_base_premium(coverage_a, county)

    # The renewal notice mentions a rate adjustment for the Building Code Effectiveness Grade,
    # [cite_start]ranging from a 12% credit to a 1% surcharge[cite: 69, 70].
    # We'll use the 'protection_class' as a proxy for this adjustment.
    def get_bceg_factor(pc):
        if pc <= 3: return 0.88 # 12% credit
        if pc <= 6: return 1.00 # Neutral
        return 1.01 # 1% surcharge
    
    bceg_factor = get_bceg_factor(prop.get("protection_class", 5))
    premium_after_bceg = base_premium * bceg_factor

    # [cite_start]The notice template also itemizes several Florida-specific fees[cite: 57, 58, 59, 60].
    # We will add flat amounts for these as an approximation.
    fhcf_fee = 75  # Florida Hurricane Catastrophe Fund
    figa_fee = 20  # Florida Insurance Guaranty Association
    citizens_fee = 15 # Citizens Property Insurance Corporation Assessment

    final_premium = premium_after_bceg + fhcf_fee + figa_fee + citizens_fee

    return {"final_premium": round(final_premium), "insurer": "Progressive / ASI"}

# ----------------------------------------------------------------------
# 🚀 Main Handler and Test Suite
# ----------------------------------------------------------------------

def calculate_all_dp3_premiums(policy_details: dict) -> dict:
    """Main handler that runs calculations for all insurers for a DP-3 policy."""
    return {
        "citizens_public": calculate_citizens_dp3_premium(policy_details),
        "universal_private": calculate_universal_dp3_premium(policy_details),
        "statefarm_private": calculate_statefarm_dp3_premium(policy_details),
        "progressive_private": calculate_progressive_dp3_premium(policy_details)
    }

def get_mean(df, zipcode):
    row = df[df['Zipcode'].astype(str) == str(zipcode)]
    return row.to_dict('records')[0]

def calculate_score_tier(score):
  """Converts a credit score to a rating number from 1 to 10.

  Args:
    score: An integer representing the credit score.

  Returns:
    An integer from 1 to 10 representing the credit rating, or None if the
    score is out of range.
  """
  if 825 <= score <= 850:
    return 10
  elif 800 <= score <= 824:
    return 9
  elif 770 <= score <= 799:
    return 8
  elif 740 <= score <= 769:
    return 7
  elif 700 <= score <= 739:
    return 6
  elif 670 <= score <= 699:
    return 5
  elif 620 <= score <= 669:
    return 4
  elif 580 <= score <= 619:
    return 3
  elif 500 <= score <= 579:
    return 2
  elif 300 <= score <= 499:
    return 1
  else:
    return None

def calculate(houseInfo, userData, df):
    override = None
    for houseOverride in userData.get('propertyAddresses',[]):
        if houseOverride['address'] == houseInfo['Street Address']:
            override = houseOverride
            break

    zipcodeRatings = get_mean(df, houseInfo['Zipcode'])
    houseValue = 0
    if houseInfo['Tax Assessed Value']:
        houseValue = houseInfo['Tax Assessed Value']
    else:
        houseValue = houseInfo['Zestimate']
        
    policyData = {
        "property": {
            "county": "Orange",
            "coverage_a_amount": houseValue,
            "construction_type": "masonry",
            "year_built": houseInfo['Year Built'],
            "roof_year": 2015, 
            "wind_mitigation_rating": zipcodeRatings['Wind Mitigation Risk Tier'],
            "protection_class": zipcodeRatings['Average ISO PPC Rating']
        },
        "policyholder": {
            "deductible": userData.get('deductible', 1000),
            "insurance_score_tier": calculate_score_tier(userData.get('creditScore', 400))
        }
    }
    if override is not None:
        policyData['property']['protection_class'] = override['isoScore']
        policyData['property']['roof_year'] = override['roofYear']
        policyData['property']['wind_mitigation_rating'] = override['windMitigationRating']
        policyData['property']['construction_type'] = override['constructionType']
        policyData['property']['coverage_a_amount'] = override['coverageAmount']

    if policyData['property']['wind_mitigation_rating'] == 1:
        policyData['property']['wind_mitigation_rating'] = 'A'
    if policyData['property']['wind_mitigation_rating'] == 2:
        policyData['property']['wind_mitigation_rating'] = 'B'
    if policyData['property']['wind_mitigation_rating'] == 3:
        policyData['property']['wind_mitigation_rating'] = 'C'
    if policyData['property']['wind_mitigation_rating'] == 4:
        policyData['property']['wind_mitigation_rating'] = 'D'

    return calculate_all_dp3_premiums(policyData)