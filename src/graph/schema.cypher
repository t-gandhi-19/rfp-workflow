// Graph schema — idempotent. Safe to re-run on every startup.
//
// Applied by `python -m scripts.apply_schema`, which runs statement by statement
// (the Neo4j driver takes one statement per call). Statements are separated by
// a line containing only `;`.
//
// The vector index dimension below is a LITERAL, because Neo4j does not accept
// a parameter inside index options. It must equal `model.dimensions` in
// config/embedding.yaml, and a test asserts exactly that — so the pin stays
// executable rather than becoming a comment nobody rechecks.

// ---------------------------------------------------------------------------
// Uniqueness constraints. Each also creates a backing index.
// ---------------------------------------------------------------------------
CREATE CONSTRAINT domain_key IF NOT EXISTS
FOR (n:Domain) REQUIRE n.key IS UNIQUE
;
CREATE CONSTRAINT customer_id IF NOT EXISTS
FOR (n:Customer) REQUIRE n.id IS UNIQUE
;
CREATE CONSTRAINT rfp_id IF NOT EXISTS
FOR (n:RFP) REQUIRE n.id IS UNIQUE
;
CREATE CONSTRAINT question_id IF NOT EXISTS
FOR (n:Question) REQUIRE n.id IS UNIQUE
;
CREATE CONSTRAINT answer_id IF NOT EXISTS
FOR (n:Answer) REQUIRE n.id IS UNIQUE
;
CREATE CONSTRAINT capability_id IF NOT EXISTS
FOR (n:Capability) REQUIRE n.id IS UNIQUE
;
CREATE CONSTRAINT sme_id IF NOT EXISTS
FOR (n:SME) REQUIRE n.id IS UNIQUE
;
CREATE CONSTRAINT outcome_value IF NOT EXISTS
FOR (n:Outcome) REQUIRE n.value IS UNIQUE
;
CREATE CONSTRAINT case_study_code IF NOT EXISTS
FOR (n:CaseStudy) REQUIRE n.code IS UNIQUE
;
CREATE CONSTRAINT product_code IF NOT EXISTS
FOR (n:Product) REQUIRE n.code IS UNIQUE
;
CREATE CONSTRAINT certification_code IF NOT EXISTS
FOR (n:Certification) REQUIRE n.code IS UNIQUE
;
CREATE CONSTRAINT location_code IF NOT EXISTS
FOR (n:Location) REQUIRE n.code IS UNIQUE
;

// Vendor codes are unique in their own right, not just vendor ids — a duplicate
// code would make entity resolution ambiguous exactly where it must not be.
CREATE CONSTRAINT vendor_code IF NOT EXISTS
FOR (n:Vendor) REQUIRE n.code IS UNIQUE
;

// ---------------------------------------------------------------------------
// Lookup indexes. Entity resolution matches on a normalised name, so the
// normalised form is what gets indexed.
// ---------------------------------------------------------------------------
CREATE INDEX vendor_normalised_name IF NOT EXISTS
FOR (n:Vendor) ON (n.normalised_name)
;
CREATE INDEX product_normalised_name IF NOT EXISTS
FOR (n:Product) ON (n.normalised_name)
;
CREATE INDEX certification_normalised_name IF NOT EXISTS
FOR (n:Certification) ON (n.normalised_name)
;
CREATE INDEX customer_normalised_name IF NOT EXISTS
FOR (n:Customer) ON (n.normalised_name)
;
CREATE INDEX location_normalised_name IF NOT EXISTS
FOR (n:Location) ON (n.normalised_name)
;
// Every registry node also carries the shared :RegistryEntity label and an
// `entity_kind`. That lets entity resolution be ONE static, parameterised query
// instead of one per label — which is what keeps "agents never write free
// Cypher" cheap to honour as entity types are added.
CREATE INDEX registry_kind_code IF NOT EXISTS
FOR (n:RegistryEntity) ON (n.entity_kind, n.code)
;
CREATE INDEX registry_kind_name IF NOT EXISTS
FOR (n:RegistryEntity) ON (n.entity_kind, n.normalised_name)
;

CREATE INDEX answer_superseded IF NOT EXISTS
FOR (n:Answer) ON (n.superseded)
;
CREATE INDEX question_domain IF NOT EXISTS
FOR (n:Question) ON (n.domain)
;

// ---------------------------------------------------------------------------
// Native vector index — hybrid retrieval happens in one query, so there is no
// second datastore to fall out of sync with the graph.
// ---------------------------------------------------------------------------
CREATE VECTOR INDEX question_embedding IF NOT EXISTS
FOR (n:Question) ON (n.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
} }
;
