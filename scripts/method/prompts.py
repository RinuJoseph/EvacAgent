#PARSING PROMPT
INFO_PARSING_PROMPT = """
You are a precise Geospatial Intent Parser for emergency evacuation navigation queries.

User Query: {query}

Extract these slots:
- anchor_location: exact phrase as mentioned; if none, "UNKNOWN"
- travel_threshold_meters: integer meters if explicitly specified; convert miles/km/feet to meters; Unknown if not specified
- target_poi_type: probability distribution over allowed POI categories
- navigation_mode: probability distribution over allowed modes

Allowed target_poi_type values (use ONLY these exact strings):
hospital, clinic, doctors, pharmacy, fire_station, police, shelter, community_centre, school

Allowed navigation_mode values (use ONLY these exact strings):
drive, walk

=== RULES FOR PROBABILITY ASSIGNMENT ===

For navigation_mode:
- If a mode is clearly mentioned ("drive to", "walk", "by car", "on foot"), assign 0.95–1.00 to that mode and 0.00 to the other.
- If no mode is mentioned at all, assign exactly 0.50 to drive and 0.50 to walk.

For target_poi_type:

A) STRONG DIRECT MATCHES (SINGLETON)
If the query clearly indicates one type, output a SINGLE high-probability category (0.95–1.00) and set all others to 0.00.

B) SPECIAL AMBIGUOUS CASE: "EMERGENCY SERVICES" (NOT MEDICAL-SPECIFIC)
If the query says "emergency services", "emergency help", "emergency assistance", 
AND it does NOT clearly indicate medical-only, police-only, or fire-only:
- DO NOT choose one single category.
- Split probability across exactly TWO categories: police, fire_station so they sum to 1.00 after rounding.
- All other categories must be 0.00.

C) AMBIGUOUS BETWEEN MEDICAL TYPES
If the query suggests medical related services or emergency but not clearly
- Use hospital, clinic, doctors, pharmacy.
- All others 0.00.

D) EVACUATION / SAFE-PLACE QUERIES
If the query is about evacuation, safe place, sheltering, where to stay, or a disaster shelter: shelter,school,community center.


E) GENERAL RULES
1. In most cases, assign 0.90–1.00 to ONE best category.
2. You may assign 0.05–0.10 total to at most ONE or TWO closely related alternatives ONLY if the query strongly suggests them.
3. All other categories that are irrelevant MUST be exactly 0.00.
4. Probabilities must sum to exactly 1.00 after rounding to two decimals.

=== FEW-SHOT EXAMPLES ===

=== OUTPUT RULES ===
- Output JSON only — no markdown, no comments, no explanations.
- Round all probabilities to exactly 2 decimal places.
- target_poi_type must sum to exactly 1.00.
- navigation_mode must sum to exactly 1.00.

Output format:
{{
  "anchor_location": "exact phrase or UNKNOWN",
  "target_poi_type": {{
    "hospital": 0.00,
    "clinic": 0.00,
    "doctors": 0.00,
    "pharmacy": 0.00,
    "fire_station": 0.00,
    "police": 0.00,
    "shelter": 0.00,
    "community_centre": 0.00,
    "school": 0.00
  }},
  "travel_threshold_meters": null,
  "navigation_mode": {{
    "drive": 0.00,
    "walk": 0.00
  }}
}}
""".strip()

#THRESHOLD SAMPLING PROMPT
THRESHOLD_SAMPLING_PROMPT = """
User query: "{query}"

Return a travel distance in METERS (integer only).

Rules:
1. EXPLICIT distance given (e.g., "within 2 miles", "5 km") → return exact conversion, no variation
2. if vague words or no distance mentioned, return a random number between 500 and 25000 meters


Output: just the number, nothing else.
"""

# Text-to-SQL
TEXT_TO_SQL_PROMPT = """
You generate ONE DuckDB Spatial SQL SELECT query from a parsed geospatial query.
Output RAW SQL only (no JSON, no comments, no prose).

Environment:
- Table: {table}(name TEXT, fclass TEXT, lat DOUBLE, long DOUBLE)
- Use ST_Distance_Spheroid(ST_Point(lat, long), ST_Point(ref_lat, ref_long)) to measure distance.

Inputs:
- parsed_query: {{ "start_location", "target_poi_type", "distance_threshold", "navigation_mode" }}
- ref_info: {{ "name", "lat", "long", "fclass" }}

Requirements:
1. If target_poi_type is known, filter by it.
2. Select columns: name, fclass, lat, long, and computed distance as euclid_distance_m.
3. Only include rows within the distance threshold.
4. Do not sort the output.
5. Use single quotes for strings and escape any internal quotes.
6. If distance_threshold is not provided, do not filter by distance.

Generate the SQL for:
parsed_query = {parsed_query_json_here}
ref_info = {ref_info_json_here}
""".strip()


# Question Generation
ASSISTANT_QUESTION_PROMPT = """You are a routing assistant. Write ONE short clarification question.
Ask about ONLY the given slot.
Do not mention coordinates formats.
Do not include options in the question.
Output ONLY a JSON object: {{"question": "..."}}

INPUT:
{payload}
"""

# User Answer Simulation
USER_ANSWER_PROMPT = """You are simulating a user answering a clarification question.
Pick EXACTLY ONE option id that matches the TRUE_INTENT.

For anchor questions:
- Use the true anchor location (true_anchor_location) from the input, NOT cluster descriptions
- Match the true anchor coordinates to the option with the closest coordinates
- The true anchor location comes directly from ground truth, not from clusters

For other questions:
- Match based on the true_intent values

Output ONLY JSON: {{"choice_id": "<id>"}}
The id MUST be exactly one of the provided option ids.

INPUT:
{payload}
"""
