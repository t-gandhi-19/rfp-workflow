"""The synthetic registry — the closed world every answer is checked against.

Every name here is fictional (CLAUDE.md rule 1). Industry-standard certification
names are real because they are public standards, not anyone's private data.

This is the source of truth for the CSVs under fixtures/registry/. Nothing in a
generated answer may name an entity that is not in this file, and a test runs
the real entity resolver over the generated corpus to prove it.
"""

from __future__ import annotations

from typing import NamedTuple


class Vendor(NamedTuple):
    code: str
    name: str
    services: str
    active: str


class Product(NamedTuple):
    code: str
    name: str
    description: str
    domain: str


class Certification(NamedTuple):
    code: str
    name: str
    scope: str


class CaseStudy(NamedTuple):
    code: str
    title: str
    customer: str
    domain: str
    publicly_usable: str
    summary: str


class Customer(NamedTuple):
    id: str
    name: str
    industry: str


class SME(NamedTuple):
    id: str
    name: str
    role: str
    capabilities: str


class Capability(NamedTuple):
    id: str
    name: str


class Location(NamedTuple):
    code: str
    name: str
    country: str
    kind: str


# ---------------------------------------------------------------------------
# 12 partner vendors
# ---------------------------------------------------------------------------
VENDORS: tuple[Vendor, ...] = (
    Vendor("VND-0001", "CloudNova Partners", "landing zone build; managed platform ops", "true"),
    Vendor("VND-0002", "Meridian Systems Integration", "application remediation; testing", "true"),
    Vendor("VND-0003", "Apex Data Movers", "database migration; replication tooling", "true"),
    Vendor("VND-0004", "Northwind Cloud Advisory", "assessment; business case", "true"),
    Vendor("VND-0005", "Falcon Ridge Networks", "network design; hybrid connectivity", "true"),
    Vendor("VND-0006", "Silverbrook Security Labs", "security review; penetration testing", "true"),
    Vendor("VND-0007", "Harbourline Consulting", "programme management; change enablement", "true"),
    Vendor("VND-0008", "Vantage Point Analytics", "FinOps analytics; showback reporting", "true"),
    Vendor("VND-0009", "Ironvale Infrastructure", "data-centre exit; hardware disposal", "true"),
    Vendor("VND-0010", "Cobalt Street Software", "application modernisation; refactoring", "true"),
    Vendor("VND-0011", "Redgate Hollow Testing", "non-functional testing; performance", "false"),
    Vendor("VND-0012", "Thornfield Managed Services", "24x7 operations; service desk", "true"),
)

# ---------------------------------------------------------------------------
# 8 internal accelerators
# ---------------------------------------------------------------------------
PRODUCTS: tuple[Product, ...] = (
    Product("PRD-0001", "MigrateHub", "discovery and assessment workbench", "cloud_migration"),
    Product("PRD-0002", "CloudLens", "dependency mapping and wave planning", "cloud_migration"),
    Product("PRD-0003", "ShiftRunner", "workload migration execution engine", "cloud_migration"),
    Product("PRD-0004", "FinScope", "cloud cost modelling and FinOps reporting", "cloud_migration"),
    Product("PRD-0005", "SecureLand", "landing-zone factory with guardrails", "cloud_migration"),
    Product("PRD-0006", "DataFerry", "database and data-estate migration", "cloud_migration"),
    Product("PRD-0007", "OpsBridge", "managed operations and observability", "cloud_migration"),
    Product(
        "PRD-0008", "TestHarbor", "migration validation and regression suite", "cloud_migration"
    ),
)

# ---------------------------------------------------------------------------
# 7 certifications. Public standards, so the names are real.
# ---------------------------------------------------------------------------
CERTIFICATIONS: tuple[Certification, ...] = (
    Certification("CRT-0001", "ISO 27001", "information security management"),
    Certification("CRT-0002", "ISO 9001", "quality management"),
    Certification("CRT-0003", "SOC 2 Type II", "security, availability and confidentiality"),
    Certification("CRT-0004", "AWS Migration Competency", "AWS migration delivery"),
    Certification("CRT-0005", "AWS Advanced Consulting Partner", "AWS partner tier"),
    Certification("CRT-0006", "Azure Expert MSP", "Azure managed services"),
    Certification("CRT-0007", "CMMI Level 5", "delivery process maturity"),
)

# ---------------------------------------------------------------------------
# 6 case studies — 4 publicly usable, 2 not
# ---------------------------------------------------------------------------
CASE_STUDIES: tuple[CaseStudy, ...] = (
    CaseStudy(
        "CST-0001",
        "Insurance core platform migration",
        "Meridian Insurance Group",
        "cloud_migration",
        "true",
        "Migrated 240 workloads to AWS across four waves with zero unplanned downtime.",
    ),
    CaseStudy(
        "CST-0002",
        "Retail data-centre exit",
        "Northfield Retail Co",
        "cloud_migration",
        "true",
        "Closed two data centres in fourteen months, retiring 900 physical hosts.",
    ),
    CaseStudy(
        "CST-0003",
        "Manufacturing ERP replatform",
        "Helvetia Manufacturing AG",
        "cloud_migration",
        "true",
        "Replatformed a regional ERP estate to Azure with a 6R disposition per workload.",
    ),
    CaseStudy(
        "CST-0004",
        "Logistics landing-zone rebuild",
        "Calder Freight Holdings",
        "cloud_migration",
        "true",
        "Rebuilt a non-compliant landing zone into a guardrailed multi-account structure.",
    ),
    CaseStudy(
        "CST-0005",
        "Health records platform migration",
        "Bluepine Health Systems",
        "cloud_migration",
        "false",
        "Migrated a regulated records platform under a customer confidentiality agreement.",
    ),
    CaseStudy(
        "CST-0006",
        "Banking DR reconstruction",
        "Kestrel Financial Partners",
        "cloud_migration",
        "false",
        "Rebuilt disaster recovery for a regulated banking estate; client name withheld.",
    ),
)

# ---------------------------------------------------------------------------
# 8 customers
# ---------------------------------------------------------------------------
CUSTOMERS: tuple[Customer, ...] = (
    Customer("CUS-0001", "Meridian Insurance Group", "insurance"),
    Customer("CUS-0002", "Northfield Retail Co", "retail"),
    Customer("CUS-0003", "Helvetia Manufacturing AG", "manufacturing"),
    Customer("CUS-0004", "Bluepine Health Systems", "healthcare"),
    Customer("CUS-0005", "Calder Freight Holdings", "logistics"),
    Customer("CUS-0006", "Kestrel Financial Partners", "financial services"),
    Customer("CUS-0007", "Larkspur Energy Utilities", "utilities"),
    Customer("CUS-0008", "Brightmoor Public Services", "public sector"),
)

#: The one customer whose material is confidential (build prompt §8).
CONFIDENTIAL_CUSTOMER = "Bluepine Health Systems"

#: The customer issuing the golden RFP.
GOLDEN_RFP_CUSTOMER = "Meridian Insurance Group"

# ---------------------------------------------------------------------------
# 6 capabilities and the 6 SMEs who own them
# ---------------------------------------------------------------------------
CAPABILITIES: tuple[Capability, ...] = (
    Capability("CAP-0001", "assessment"),
    Capability("CAP-0002", "landing-zones"),
    Capability("CAP-0003", "workload-migration"),
    Capability("CAP-0004", "data-migration"),
    Capability("CAP-0005", "security-compliance"),
    Capability("CAP-0006", "finops-operations"),
)

SMES: tuple[SME, ...] = (
    SME("SME-0001", "Priya Raghunathan", "Principal Migration Architect", "CAP-0001|CAP-0003"),
    SME("SME-0002", "Tomas Lindqvist", "Landing Zone Lead", "CAP-0002"),
    SME("SME-0003", "Adaeze Okonkwo", "Data Migration Lead", "CAP-0004"),
    SME("SME-0004", "Marcus Feldman", "Security and Compliance Lead", "CAP-0005"),
    SME("SME-0005", "Yuki Tanaka", "FinOps and Operations Lead", "CAP-0006"),
    SME("SME-0006", "Rosa Delgado", "Delivery Director", "CAP-0001|CAP-0006"),
)

# ---------------------------------------------------------------------------
# Delivery locations.
#
# Not in the build prompt's registry list, but §15 requires every named location
# to resolve against the registry, and company_info answers legitimately name
# delivery centres. Without this file those answers would hard-fail grounding.
# ---------------------------------------------------------------------------
LOCATIONS: tuple[Location, ...] = (
    Location("LOC-0001", "Manchester", "United Kingdom", "delivery centre"),
    Location("LOC-0002", "Kraków", "Poland", "delivery centre"),
    Location("LOC-0003", "Pune", "India", "delivery centre"),
    Location("LOC-0004", "Lisbon", "Portugal", "delivery centre"),
    Location("LOC-0005", "Toronto", "Canada", "delivery centre"),
    Location("LOC-0006", "Frankfurt", "Germany", "data residency region"),
    Location("LOC-0007", "Dublin", "Ireland", "data residency region"),
)


def capability_id_by_name(name: str) -> str:
    for capability in CAPABILITIES:
        if capability.name == name:
            return capability.id
    raise KeyError(f"unknown capability: {name}")


def sme_for_capability(capability_id: str) -> SME:
    """First SME owning a capability, by id — deterministic."""
    for sme in sorted(SMES, key=lambda s: s.id):
        if capability_id in sme.capabilities.split("|"):
            return sme
    raise KeyError(f"no SME owns capability {capability_id}")
