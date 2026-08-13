"""The 40 synthetic Q&A topics (build prompt §8).

Each topic supplies its own substance; the generator wraps it in shared
governance, evidence and delivery scaffolding chosen deterministically. That
keeps every answer in the 150-300 word band and citing real registry entities,
without 8,000 words of hand-written prose drifting out of sync with the registry.

Four subjects appear twice, as a `-v1` superseded by a `-v2`. Those are the
supersession chains: the v1 answers name something retired, the v2 answers name
what replaced it, and retrieval must surface the v2.

No pricing figure appears anywhere in this file. Commercial topics cover process
only (build prompt §8).
"""

from __future__ import annotations

from typing import NamedTuple


class Topic(NamedTuple):
    key: str
    question_type: str
    capability: str
    question: str
    opening: str
    method: str
    products: tuple[str, ...] = ()
    vendors: tuple[str, ...] = ()
    certifications: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    #: Key of the topic that supersedes this one, if any.
    superseded_by: str | None = None
    #: Which SUBJECT this topic is about, as opposed to which record it is.
    #:
    #: `key` is an identity — it is unique, and the golden RFP source joins to a
    #: corpus answer through it. `family` is a grouping, and the two are not the
    #: same thing: `landing-zone-v1` and `landing-zone-v2` are two records of one
    #: subject. Calibration needs the grouping, because "genuinely the same
    #: question" is a statement about subject, not about row identity.
    #:
    #: Defaults to `key`, which is correct for the 32 topics that appear once.
    family: str | None = None


def family_of(topic: Topic) -> str:
    """The subject a topic belongs to. Its own key unless it is a version."""
    return topic.family or topic.key


TOPICS: tuple[Topic, ...] = (
    # ---------------- technical (18) ----------------
    Topic(
        "sixr-strategy",
        "technical",
        "assessment",
        "Describe how you determine the migration strategy (6R disposition) for each workload.",
        "Every workload receives an explicit 6R disposition — rehost, replatform, refactor, "
        "repurchase, retire or retain — before it enters a migration wave. The disposition is "
        "recorded as a decision with a named owner, so a later question about why a workload "
        "moved the way it did has an answer.",
        "Discovery runs through MigrateHub, which collects utilisation, licensing and support "
        "status, while CloudLens resolves the dependency graph so that disposition decisions "
        "account for what a workload talks to rather than the workload alone. Northwind Cloud "
        "Advisory reviews the resulting dispositions with application owners before the plan is "
        "baselined.",
        products=("MigrateHub", "CloudLens"),
        vendors=("Northwind Cloud Advisory",),
    ),
    Topic(
        "dependency-discovery",
        "technical",
        "assessment",
        "What is your approach to application dependency discovery?",
        "Dependency discovery is agentless first and agent-based only where the estate is opaque. "
        "We combine network flow observation with configuration analysis, because either alone "
        "produces a map that looks complete and is not.",
        "CloudLens ingests flow data and builds a dependency graph that is reviewed with each "
        "application owner rather than accepted as generated. Findings that contradict the "
        "customer's own documentation are raised explicitly, since an undocumented dependency "
        "discovered at cutover is the most common cause of an extended outage window.",
        products=("CloudLens", "MigrateHub"),
    ),
    Topic(
        "landing-zone-v1",
        "technical",
        "landing-zones",
        "Describe your landing zone design and how guardrails are enforced.",
        "Our landing zone establishes account separation, network segmentation and a baseline "
        "control set before any workload lands. Controls are applied as code so that drift is "
        "detectable rather than discovered during an audit.",
        "The zone is assembled with our LandingForge templates and hardened by Silverbrook "
        "Security Labs before handover. Preventive guardrails block non-compliant resource "
        "creation outright, and detective controls report drift daily.",
        products=(),
        vendors=("Silverbrook Security Labs",),
        superseded_by="landing-zone-v2",
        family="landing-zone",
    ),
    Topic(
        "landing-zone-v2",
        "technical",
        "landing-zones",
        "Describe your landing zone design and how guardrails are enforced.",
        "Our landing zone establishes account separation, network segmentation and a baseline "
        "control set before any workload lands. Controls are applied as code, so drift is "
        "detectable continuously rather than discovered during an audit.",
        "SecureLand builds the multi-account structure from a versioned factory definition, with "
        "preventive guardrails that block non-compliant resource creation outright and detective "
        "controls that report drift daily. Silverbrook Security Labs reviews the control set "
        "before handover, and CloudNova Partners operates it thereafter.",
        products=("SecureLand",),
        vendors=("Silverbrook Security Labs", "CloudNova Partners"),
        family="landing-zone",
    ),
    Topic(
        "cutover-rollback-v1",
        "technical",
        "workload-migration",
        "How do you plan cutover and what is your rollback position?",
        "Each wave has a written cutover runbook with timed steps, named owners and explicit "
        "go/no-go gates. Rollback is planned as a first-class path, not as an exception.",
        "Runbooks are rehearsed in a non-production dress rehearsal before the live window. "
        "Rollback is achieved by keeping the source environment warm and reversing DNS, with the "
        "decision point placed before the point of no return.",
        products=(),
        vendors=("Meridian Systems Integration",),
        superseded_by="cutover-rollback-v2",
        family="cutover-rollback",
    ),
    Topic(
        "cutover-rollback-v2",
        "technical",
        "workload-migration",
        "How do you plan cutover and what is your rollback position?",
        "Each wave has a written cutover runbook with timed steps, named owners and explicit "
        "go/no-go gates. Rollback is a first-class path that is rehearsed, not an exception that "
        "is improvised.",
        "ShiftRunner executes the runbook so each step is timestamped and auditable, and "
        "TestHarbor runs the validation suite at every gate. The source environment stays warm "
        "until the post-cutover validation gate passes, so rollback remains available rather "
        "than theoretical. Meridian Systems Integration staffs the application-side checks.",
        products=("ShiftRunner", "TestHarbor"),
        vendors=("Meridian Systems Integration",),
        family="cutover-rollback",
    ),
    Topic(
        "database-migration-v1",
        "technical",
        "data-migration",
        "How do you migrate production databases with minimal downtime?",
        "Database migration is planned around the tolerance for downtime that the business "
        "actually states, rather than a default assumption of near-zero.",
        "We use a backup-and-restore approach for systems with a tolerant window, scheduled "
        "during an agreed outage. Apex Data Movers validates row counts and checksums after "
        "restore, and the source remains available for comparison for one week.",
        products=(),
        vendors=("Apex Data Movers",),
        superseded_by="database-migration-v2",
        family="database-migration",
    ),
    Topic(
        "database-migration-v2",
        "technical",
        "data-migration",
        "How do you migrate production databases with minimal downtime?",
        "Database migration is planned around the downtime tolerance the business actually "
        "states. Where that tolerance is small, we use continuous replication and cut over "
        "inside a short window rather than moving the whole estate at once.",
        "DataFerry performs schema conversion and continuous change-data-capture replication, "
        "with lag monitored against a threshold that must be met before the cutover gate opens. "
        "Apex Data Movers validates row counts and checksums on both sides, and the source stays "
        "readable for a reconciliation period after cutover.",
        products=("DataFerry", "TestHarbor"),
        vendors=("Apex Data Movers",),
        family="database-migration",
    ),
    Topic(
        "network-connectivity",
        "technical",
        "landing-zones",
        "Describe your approach to hybrid network connectivity during migration.",
        "Hybrid connectivity is established and load-tested before the first workload moves, "
        "because discovering a bandwidth ceiling mid-wave stops the programme rather than "
        "slowing it.",
        "Falcon Ridge Networks designs redundant private connectivity with a documented failover "
        "path, and we validate throughput against the largest planned replication load rather "
        "than an average. Routing and DNS changes are staged so that each wave can be reversed "
        "independently.",
        vendors=("Falcon Ridge Networks",),
    ),
    Topic(
        "disaster-recovery",
        "technical",
        "workload-migration",
        "How is disaster recovery designed and tested for migrated workloads?",
        "Recovery objectives are agreed per workload tier before design, and the design is then "
        "tested against those objectives rather than asserted to meet them.",
        "Recovery is rebuilt in the target using infrastructure as code so the recovery "
        "environment is reproducible. TestHarbor drives the recovery test and records measured "
        "recovery times, which are reported against the agreed objectives. Failures produce a "
        "corrective action with an owner.",
        products=("TestHarbor", "SecureLand"),
    ),
    Topic(
        "wave-planning",
        "technical",
        "assessment",
        "How do you sequence workloads into migration waves?",
        "Waves are sequenced by dependency cluster and business risk, so that tightly coupled "
        "systems move together and the earliest waves carry the least business exposure.",
        "CloudLens groups workloads by their dependency graph, and MigrateHub overlays business "
        "criticality and change-freeze calendars. The first wave is deliberately chosen to be "
        "recoverable, so the programme learns on workloads where a mistake is affordable.",
        products=("CloudLens", "MigrateHub"),
    ),
    Topic(
        "application-remediation",
        "technical",
        "workload-migration",
        "How do you handle applications that cannot be rehosted without change?",
        "Applications that will not lift cleanly are identified during assessment, not at "
        "cutover. Each receives a remediation plan sized before the wave is committed.",
        "Cobalt Street Software performs the remediation work — configuration externalisation, "
        "hard-coded endpoint removal and library upgrades — and TestHarbor establishes the "
        "regression baseline before any change is made, so remediation defects are "
        "distinguishable from pre-existing ones.",
        products=("TestHarbor",),
        vendors=("Cobalt Street Software",),
    ),
    Topic(
        "data-centre-exit",
        "technical",
        "workload-migration",
        "Describe your approach to decommissioning the source data centre.",
        "Decommissioning is a tracked workstream with its own gates, because an estate that "
        "migrates but never shuts down delivers no saving.",
        "Ironvale Infrastructure handles physical decommissioning and certified disposal once a "
        "workload has passed its post-migration soak period. MigrateHub tracks each asset from "
        "migrated to powered-off to removed, so the exit position is reportable at any point.",
        products=("MigrateHub",),
        vendors=("Ironvale Infrastructure",),
    ),
    Topic(
        "observability",
        "technical",
        "finops-operations",
        "What observability do you establish for migrated workloads?",
        "Observability is established before cutover, not after, so that post-migration "
        "behaviour can be compared against a pre-migration baseline.",
        "OpsBridge collects metrics, logs and traces into a single view with alerting mapped to "
        "the service model. Thornfield Managed Services operates the alerting rota. Baselines "
        "captured before the move are retained so a performance complaint can be tested rather "
        "than debated.",
        products=("OpsBridge",),
        vendors=("Thornfield Managed Services",),
    ),
    Topic(
        "performance-validation",
        "technical",
        "workload-migration",
        "How do you validate performance after migration?",
        "Performance is validated against a measured pre-migration baseline, using the "
        "customer's own transactions rather than synthetic load alone.",
        "TestHarbor runs the regression and non-functional suites at each gate, and Redgate "
        "Hollow Testing supplies specialist performance engineering where a workload has an "
        "unusual profile. Results are reported per workload, and a workload that fails its "
        "threshold does not pass its gate.",
        products=("TestHarbor",),
        vendors=("Redgate Hollow Testing",),
    ),
    Topic(
        "identity-integration",
        "technical",
        "security-compliance",
        "How is identity and access management integrated in the target platform?",
        "Identity federates to the customer's existing provider rather than creating a parallel "
        "directory, so joiners and leavers continue to flow through one process.",
        "SecureLand provisions role definitions per account with least-privilege baselines, and "
        "Silverbrook Security Labs reviews the role set before production access is granted. "
        "Break-glass accounts exist, are monitored, and their use raises an alert.",
        products=("SecureLand",),
        vendors=("Silverbrook Security Labs",),
    ),
    Topic(
        "encryption-key-management",
        "technical",
        "security-compliance",
        "Describe encryption and key management in the target environment.",
        "Data is encrypted in transit and at rest by default, with key custody agreed explicitly "
        "because it determines who can be compelled to produce data.",
        "SecureLand enforces encryption as a preventive guardrail, so an unencrypted store "
        "cannot be created rather than being detected later. Customer-managed keys are supported "
        "where the customer wishes to retain custody, and Silverbrook Security Labs validates "
        "the key rotation schedule.",
        products=("SecureLand",),
        vendors=("Silverbrook Security Labs",),
        certifications=("ISO 27001",),
    ),
    Topic(
        "migration-tooling-selection",
        "technical",
        "assessment",
        "How do you select migration tooling for a given estate?",
        "Tooling is selected per workload class against the constraints that actually bind — "
        "downtime tolerance, licensing and operating system support — rather than standardising "
        "on one tool for the whole estate.",
        "ShiftRunner covers the bulk rehost path, DataFerry covers the data estate, and native "
        "provider services are used where they fit better. The selection is recorded per "
        "workload class in the migration plan with the reason attached.",
        products=("ShiftRunner", "DataFerry", "MigrateHub"),
    ),
    # ---------------- compliance (8) ----------------
    Topic(
        "iso-soc-scope-v1",
        "compliance",
        "security-compliance",
        "What security certifications do you hold and what is their scope?",
        "We hold certifications covering our delivery organisation and the managed services we "
        "operate on behalf of customers.",
        "Our ISO 27001 certification covers information security management across delivery "
        "operations, and ISO 9001 covers quality management. Certificates and current scope "
        "statements are provided on request.",
        certifications=("ISO 27001", "ISO 9001"),
        superseded_by="iso-soc-scope-v2",
        family="iso-soc-scope",
    ),
    Topic(
        "iso-soc-scope-v2",
        "compliance",
        "security-compliance",
        "What security certifications do you hold and what is their scope?",
        "We hold certifications covering our delivery organisation, our managed services, and "
        "our cloud partner status. Scope statements are provided rather than summarised, because "
        "a certification's scope matters more than its existence.",
        "ISO 27001 covers information security management across delivery operations; SOC 2 Type "
        "II covers security, availability and confidentiality for managed services; ISO 9001 "
        "covers quality management. We are an AWS Advanced Consulting Partner holding the AWS "
        "Migration Competency, and an Azure Expert MSP.",
        certifications=(
            "ISO 27001",
            "SOC 2 Type II",
            "ISO 9001",
            "AWS Advanced Consulting Partner",
            "AWS Migration Competency",
            "Azure Expert MSP",
        ),
        family="iso-soc-scope",
    ),
    Topic(
        "data-residency",
        "compliance",
        "security-compliance",
        "How do you guarantee data residency requirements are met?",
        "Residency requirements are captured as explicit constraints during design and then "
        "enforced technically, because a policy statement alone does not prevent a deployment "
        "into the wrong region.",
        "SecureLand restricts the permitted region list per account, so a resource outside the "
        "agreed geography cannot be created. Where residency is constrained to the EU we use "
        "Frankfurt or Dublin, and the constraint is recorded in the landing-zone definition "
        "under version control.",
        products=("SecureLand",),
        locations=("Frankfurt", "Dublin"),
        certifications=("ISO 27001",),
    ),
    Topic(
        "gdpr-handling",
        "compliance",
        "security-compliance",
        "How do you handle personal data and GDPR obligations during migration?",
        "Personal data is identified during assessment and handled under a documented processing "
        "basis. Migration does not change the customer's controller position, and our role as "
        "processor is set out in the agreement.",
        "MigrateHub records which workloads carry personal data so that handling requirements "
        "follow the workload through every wave. Silverbrook Security Labs reviews transfer "
        "paths, and test environments use masked data rather than production extracts.",
        products=("MigrateHub",),
        vendors=("Silverbrook Security Labs",),
        certifications=("ISO 27001",),
    ),
    Topic(
        "audit-rights",
        "compliance",
        "security-compliance",
        "What audit rights and evidence do you provide to customers?",
        "Customers receive evidence continuously rather than only at audit time, and audit "
        "rights are agreed in the contract with a defined notice period and scope.",
        "OpsBridge produces control evidence and access logs on a reporting schedule. Our SOC 2 "
        "Type II report is available under NDA, and ISO 27001 certificates with scope statements "
        "are provided on request. Findings raised during a customer audit are tracked to closure "
        "with named owners.",
        products=("OpsBridge",),
        certifications=("SOC 2 Type II", "ISO 27001"),
    ),
    Topic(
        "vulnerability-management",
        "compliance",
        "security-compliance",
        "Describe your vulnerability and patch management process.",
        "Vulnerabilities are triaged against severity and exposure, with remediation timescales "
        "agreed per severity band before service commencement.",
        "OpsBridge collects scan output and tracks remediation to closure. Silverbrook Security "
        "Labs performs periodic penetration testing, and findings enter the same tracked queue "
        "as scanner output so nothing is closed informally.",
        products=("OpsBridge",),
        vendors=("Silverbrook Security Labs",),
        certifications=("ISO 27001",),
    ),
    Topic(
        "access-control-review",
        "compliance",
        "security-compliance",
        "How is privileged access governed and reviewed?",
        "Privileged access is time-bound and reviewed on a fixed cycle, with every elevation "
        "recorded against a named individual and a reason.",
        "SecureLand provisions least-privilege roles per account, and OpsBridge retains the "
        "access log for the agreed retention period. Quarterly reviews are performed jointly "
        "with the customer's security function, and unused privileges are removed rather than "
        "carried forward.",
        products=("SecureLand", "OpsBridge"),
        certifications=("SOC 2 Type II",),
    ),
    Topic(
        "incident-notification",
        "compliance",
        "security-compliance",
        "What is your security incident notification process?",
        "Incidents are classified on a published severity scale, and notification timescales are "
        "agreed per severity in the service agreement rather than left to judgement.",
        "OpsBridge raises and tracks the incident record. Thornfield Managed Services runs the "
        "24x7 rota that performs initial triage, and a post-incident review with root cause and "
        "corrective actions is issued for every high-severity incident.",
        products=("OpsBridge",),
        vendors=("Thornfield Managed Services",),
        certifications=("ISO 27001",),
    ),
    # ---------------- company_info (8) ----------------
    Topic(
        "team-model",
        "company_info",
        "assessment",
        "Describe the team model you would deploy for a migration of this size.",
        "We deploy a core team that stays with the programme end to end, supplemented by "
        "specialist pods that join for the waves where their skills are needed.",
        "The core team covers architecture, migration engineering and programme management. "
        "Harbourline Consulting provides programme management and change enablement, and "
        "Northwind Cloud Advisory supports the assessment phase. Named leads are identified at "
        "contract award and do not rotate without customer agreement.",
        vendors=("Harbourline Consulting", "Northwind Cloud Advisory"),
    ),
    Topic(
        "delivery-centres",
        "company_info",
        "assessment",
        "Where are your delivery centres located and how do you cover time zones?",
        "Delivery is distributed across centres chosen to give overlapping working hours with "
        "the customer rather than pure cost arbitrage.",
        "Our principal delivery centres are Manchester, Kraków, Pune, Lisbon and Toronto. "
        "Cutover windows are staffed by the centre whose working day covers the window, with a "
        "documented handover between centres so context is not lost at a shift boundary.",
        locations=("Manchester", "Kraków", "Pune", "Lisbon", "Toronto"),
    ),
    Topic(
        "partnerships",
        "company_info",
        "assessment",
        "What cloud provider partnerships and competencies do you hold?",
        "Our partner status reflects assessed delivery capability rather than purchased tiers.",
        "We are an AWS Advanced Consulting Partner and hold the AWS Migration Competency, which "
        "requires validated customer references. On Azure we are an Azure Expert MSP, which "
        "requires an annual third-party audit of our managed service.",
        certifications=(
            "AWS Advanced Consulting Partner",
            "AWS Migration Competency",
            "Azure Expert MSP",
        ),
    ),
    Topic(
        "subcontractor-model",
        "company_info",
        "assessment",
        "Do you use subcontractors, and how are they governed?",
        "We use a small, stable panel of specialist partners rather than open-market "
        "subcontracting, and every partner is named in the proposal.",
        "CloudNova Partners, Meridian Systems Integration and Apex Data Movers are our principal "
        "delivery partners. Each operates under our quality system, is covered by the same "
        "security obligations, and is subject to the same review gates as our own teams.",
        vendors=("CloudNova Partners", "Meridian Systems Integration", "Apex Data Movers"),
        certifications=("ISO 9001",),
    ),
    Topic(
        "quality-system",
        "company_info",
        "assessment",
        "Describe your quality management system and delivery maturity.",
        "Delivery follows a documented quality system that is externally certified and internally "
        "audited, so process adherence is evidenced rather than asserted.",
        "We are certified to ISO 9001 for quality management and assessed at CMMI Level 5 for "
        "delivery process maturity. Programme artefacts follow standard templates, and gate "
        "reviews are recorded with decisions and owners.",
        certifications=("ISO 9001", "CMMI Level 5"),
    ),
    Topic(
        "reference-experience",
        "company_info",
        "assessment",
        "Describe comparable migration programmes you have delivered.",
        "We have delivered enterprise migration programmes across regulated and non-regulated "
        "sectors, at a scale comparable to this requirement.",
        "For Meridian Insurance Group we migrated 240 workloads to AWS across four waves with no "
        "unplanned downtime. For Northfield Retail Co we closed two data centres in fourteen "
        "months. For Helvetia Manufacturing AG we replatformed a regional ERP estate to Azure "
        "with a 6R disposition recorded per workload.",
    ),
    Topic(
        "knowledge-transfer",
        "company_info",
        "workload-migration",
        "How do you transfer knowledge to the customer's own teams?",
        "Knowledge transfer is a scheduled workstream with its own acceptance criteria, not a "
        "handover document produced at the end.",
        "Customer engineers work alongside our teams from the first wave. OpsBridge runbooks are "
        "written jointly and rehearsed by customer staff before service transition. Harbourline "
        "Consulting runs the change enablement programme and measures readiness rather than "
        "attendance.",
        products=("OpsBridge",),
        vendors=("Harbourline Consulting",),
    ),
    Topic(
        "retention-continuity",
        "company_info",
        "assessment",
        "How do you manage key-person risk and team continuity?",
        "Continuity is managed by design: every named role has an identified deputy who is "
        "active on the programme rather than nominal.",
        "Programme documentation is maintained in the customer's own repository so context does "
        "not leave with an individual. Rosa Delgado, our Delivery Director, holds accountability "
        "for continuity and reports staffing changes at the programme board.",
    ),
    # ---------------- commercial, process only (6) ----------------
    Topic(
        "engagement-model",
        "commercial",
        "finops-operations",
        "Describe your engagement model and how work is structured.",
        "Engagements are structured in phases with a decision gate between each, so the customer "
        "can stop, change scope or change supplier at a defined point rather than being "
        "committed to the whole programme at signature.",
        "Assessment is delivered first as a discrete phase producing a wave plan and dispositions. "
        "Migration then proceeds wave by wave, each with its own acceptance criteria. Managed "
        "operations begin per workload at service transition rather than at programme end.",
        products=("MigrateHub",),
    ),
    Topic(
        "governance-model",
        "commercial",
        "finops-operations",
        "Describe the governance structure you would put in place.",
        "Governance operates at three levels — programme board, delivery board and working "
        "group — each with a defined membership, cadence and decision rights.",
        "The programme board meets monthly and owns scope and escalation. The delivery board "
        "meets weekly and owns wave readiness and gate decisions. Harbourline Consulting "
        "provides the programme management office. Decisions are recorded with owners, and "
        "escalation paths are agreed before commencement rather than during the first dispute.",
        vendors=("Harbourline Consulting",),
    ),
    Topic(
        "raci-model",
        "commercial",
        "finops-operations",
        "Provide the RACI model for the migration programme.",
        "A RACI is agreed per workstream at mobilisation and reviewed at each phase gate, because "
        "an accountability model that is not revisited stops matching how the programme actually "
        "runs.",
        "We are accountable for migration execution and consulted on business acceptance. The "
        "customer is accountable for business acceptance and application-owner availability. "
        "Where a partner such as Meridian Systems Integration performs remediation, it is "
        "responsible for that task and we remain accountable for the outcome.",
        vendors=("Meridian Systems Integration",),
    ),
    # QA-0039. This pair exists to exercise the legal-adjacent path end to end:
    # the question IS retrieved and drafted normally, and the resulting draft is
    # then ALWAYS escalated by the forbidden-term guardrail because "warranty" is
    # in the legal term list. Contrast with pricing, which is never drafted at
    # all. If this pair is ever removed, that path loses its only exercise.
    Topic(
        "warranty-process",
        "commercial",
        "finops-operations",
        "Describe your post-migration warranty process.",
        "Each workload enters a defined warranty period beginning at its own cutover, not at "
        "programme completion, so warranty tracks the workload rather than the calendar.",
        "During warranty, defects attributable to the migration are corrected by the migration "
        "team rather than routed through the service desk. TestHarbor evidence from the cutover "
        "gate establishes the baseline used to determine attribution. Exit from warranty requires "
        "a written acceptance per workload.",
        products=("TestHarbor",),
    ),
    Topic(
        "change-control",
        "commercial",
        "finops-operations",
        "How is scope change managed during the programme?",
        "Change is managed through a single written process with impact assessed on schedule, "
        "risk and resource before any decision is taken.",
        "Change requests are raised into the delivery board, assessed within an agreed turnaround, "
        "and either approved, rejected or deferred with the reason recorded. Harbourline "
        "Consulting maintains the change log, and no change is implemented before written "
        "approval.",
        vendors=("Harbourline Consulting",),
    ),
    Topic(
        "finops-reporting",
        "commercial",
        "finops-operations",
        "How do you report cloud consumption and support cost governance?",
        "Consumption is reported against the business case from the first wave, so that variance "
        "is visible while it can still be acted on.",
        "FinScope produces showback by application and business unit, with tagging enforced at "
        "the landing zone so untagged resources cannot accumulate unattributed. Vantage Point "
        "Analytics supports the monthly review, and optimisation actions are tracked with named "
        "owners and target dates.",
        products=("FinScope", "SecureLand"),
        vendors=("Vantage Point Analytics",),
    ),
)


# ---------------------------------------------------------------------------
# Shared scaffolding. Selected by index so the corpus is reproducible.
# ---------------------------------------------------------------------------

GOVERNANCE_SENTENCES: tuple[str, ...] = (
    "Progress and exceptions are reported through the weekly delivery board, where {sme_role} "
    "holds accountability for this area. Gate decisions are recorded with named owners and a "
    "date, so a question raised months later about why a particular choice was made has an "
    "answer rather than a recollection.",
    "This workstream is owned by {sme_role}, who reports readiness at each wave gate. An "
    "unresolved exception blocks the gate rather than being carried forward into the next wave, "
    "because deferred exceptions accumulate silently and surface at cutover, which is the worst "
    "possible moment to discover them.",
    "{sme_role} owns this area and reviews it at every phase gate. Decisions, the options "
    "considered and the reason for the choice are recorded in the programme log, which is held "
    "in the customer's own repository so the reasoning survives any change of personnel on "
    "either side.",
)

EVIDENCE_SENTENCES: tuple[str, ...] = (
    "The approach falls within the scope of our {certification} certification, and the "
    "supporting artefacts are available for customer review on request. We would rather be "
    "asked for evidence than assert compliance without it.",
    "This activity is covered by the scope of our {certification} certification. Audit evidence "
    "is produced on the agreed reporting schedule rather than assembled retrospectively when an "
    "auditor asks for it.",
    "Our {certification} certification covers this activity, and the associated controls are "
    "tested as part of our internal audit cycle. Findings are tracked to closure with named "
    "owners and target dates rather than noted and forgotten.",
)

CLOSING_SENTENCES: tuple[str, ...] = (
    "Delivery is staffed from our {location} centre, with documented handover between time zones "
    "so that context is not lost at a shift boundary. Named leads are identified at contract "
    "award and do not change without customer agreement.",
    "The work is performed from our {location} delivery centre. Staffing is confirmed at "
    "mobilisation, and any subsequent change is raised at the programme board rather than made "
    "quietly.",
    "Our {location} centre provides the primary staffing, supported by the wider delivery "
    "network where wave volumes require it. Resource ramp-up is agreed in the mobilisation plan "
    "so that capacity is committed before it is needed.",
)

#: A fourth shared block, so composed answers reach the 150-300 word band without
#: 8,000 words of hand-authored prose. Each names what could go wrong, which is
#: what makes a synthetic answer read like a real one.
RISK_SENTENCES: tuple[str, ...] = (
    "The most common failure mode we see here is a dependency or constraint that nobody "
    "documented, discovered when the change window has already opened. We plan for that by "
    "validating against the running estate rather than against the customer's documentation "
    "alone.",
    "Where this has gone wrong on other programmes, the cause has usually been an assumption "
    "carried forward without being retested after scope changed. Each assumption is therefore "
    "logged with an owner and revisited at the next gate.",
    "The risk we manage most actively here is optimistic sequencing: committing a wave before "
    "its prerequisites are genuinely complete. Readiness is evidenced against written criteria, "
    "and a wave that is not ready moves rather than proceeding on optimism.",
)


#: Two alternate phrasings per topic FAMILY — the same question as a different
#: customer's RFP would ask it.
#:
#: These exist because calibration needs to measure what "genuinely the same
#: question" scores, and until they were added the corpus could not answer that.
#: Its only same-subject pairs were the four supersession chains, whose question
#: text is byte-identical, so the anchor would have been measured on exact
#: duplicates — the easiest possible case. A floor derived from that lands above
#: real paraphrase matches and rejects them.
#:
#: Keyed by FAMILY, not by key: `landing-zone-v1` and `landing-zone-v2` ask the
#: same question, so they share one set rather than getting two.
#:
#: THEY KEEP THE DOMAIN VOCABULARY, and that is a correction rather than
#: laziness. The first version of this block deliberately avoided the subject
#: noun — asking about "the target cloud foundation" instead of "the landing
#: zone" — and 51 of the 72 shared no content word at all with their original.
#: That is not how a real RFP rephrases a question; it is an adversarial lower
#: bound. Measured against it, `same_topic_p05` would sit far below what a
#: genuine match actually scores, dragging the derived floor down and making it
#: too permissive — the mirror image of the duplicate-anchor failure.
#:
#: So: same subject nouns, different verbs, framing and emphasis. Recognisably
#: the same question, demonstrably not the same sentence.
PARAPHRASES: dict[str, tuple[str, str, str]] = {
    "access-control-review": (
        "Set out the controls over privileged access and the cycle on which it is reviewed.",
        "Who approves privileged access to our systems, and what record is kept when it is "
        "granted or revoked?",
        "How is privileged access to customer environments requested, approved and "
        "periodically recertified?",
    ),
    "application-remediation": (
        "What is your approach for applications that need remediation before they can be rehosted?",
        "Where an application cannot move unchanged, how is the necessary remediation scoped "
        "and delivered?",
        "How much application remediation do you expect across an estate like ours, and who "
        "carries out the changes?",
    ),
    "audit-rights": (
        "What audit rights would we hold under contract, and what evidence would you make "
        "available?",
        "Set out the evidence you supply to support a customer audit and the access rights "
        "that accompany it.",
        "How often may we exercise our audit rights, and what evidence pack is produced for "
        "each audit?",
    ),
    "change-control": (
        "What is your change control process when programme scope moves?",
        "How are changes to agreed scope raised, assessed and approved during delivery?",
        "Describe the change control board and how a scope variation reaches a decision.",
    ),
    "cutover-rollback": (
        "How is each cutover planned, and what rollback options remain once it has begun?",
        "Set out cutover sequencing and the rollback position you hold at each go/no-go gate.",
        "What does a cutover runbook contain, and at what point does rollback stop being "
        "available?",
    ),
    "data-centre-exit": (
        "How is the source data centre decommissioned once its workloads have moved?",
        "What is your data centre exit process, including decommissioning of the remaining estate?",
        "Who owns data centre decommissioning once migration completes, and what evidence of "
        "asset disposal is provided?",
    ),
    "data-residency": (
        "What controls guarantee that data residency obligations are met and stay met?",
        "How is data residency enforced, and how would you evidence compliance to a regulator?",
        "Where will our data physically reside, and what technical controls prevent it leaving "
        "that region?",
    ),
    "database-migration": (
        "What is your method for migrating production databases while keeping downtime minimal?",
        "How are production database migrations sequenced, and what downtime should we plan for?",
        "Which database migration patterns do you use, and how is data integrity verified after "
        "cutover?",
    ),
    "delivery-centres": (
        "Which delivery centres would staff this work, and how is time zone coverage arranged?",
        "List your delivery centre locations and explain how coverage is maintained across "
        "time zones.",
        "From which delivery centres would this programme be resourced, and what working hours "
        "does that cover?",
    ),
    "dependency-discovery": (
        "How is application dependency discovery carried out, and how are the findings validated?",
        "Set out the discovery process used to map dependencies between applications before "
        "migration.",
        "What tooling discovers application dependencies, and how are gaps in the resulting map "
        "closed?",
    ),
    "disaster-recovery": (
        "What disaster recovery design applies to migrated workloads, and how often is it tested?",
        "Set out your disaster recovery approach after migration, including the testing regime "
        "and recovery targets.",
        "What recovery point and recovery time objectives do migrated workloads achieve, and "
        "how is disaster recovery rehearsed?",
    ),
    "encryption-key-management": (
        "What encryption applies in the target environment, and who holds and rotates the keys?",
        "Set out your key management model for the target platform, including encryption at "
        "rest and in transit.",
        "Which encryption standards apply to migrated data, and how is key management separated "
        "from platform administration?",
    ),
    "engagement-model": (
        "What engagement model do you propose, and how would the work be structured into phases?",
        "How is an engagement of this kind structured commercially, and where are the decision "
        "points?",
        "What does your engagement model look like in practice, from mobilisation through to "
        "steady state?",
    ),
    "finops-reporting": (
        "What visibility do we get of cloud consumption, and how does that feed cost governance?",
        "How is cloud spend reported to us, and what governance surrounds ongoing cost management?",
        "Which forums own cost governance, and how frequently is cloud consumption reported "
        "against budget?",
    ),
    "gdpr-handling": (
        "What controls apply to personal data during migration, and how are GDPR obligations "
        "discharged?",
        "How are GDPR responsibilities allocated and evidenced where personal data moves "
        "between environments?",
        "Who acts as processor for personal data during migration, and what GDPR safeguards "
        "apply to that processing?",
    ),
    "governance-model": (
        "What governance structure would you establish, and who sits on it?",
        "Set out the programme governance you propose, including forums, cadence and decision "
        "rights.",
        "Which governance forums would run this programme, at what cadence, and with what "
        "escalation route?",
    ),
    "identity-integration": (
        "How does existing identity provision connect to the target platform, and what access "
        "management model applies?",
        "Set out the identity federation and access management design for the migrated estate.",
        "How is identity federated into the target platform, and what access management "
        "controls follow from it?",
    ),
    "incident-notification": (
        "How would a security incident be notified to us, and within what timescales?",
        "Set out your incident notification process, including escalation paths for a confirmed "
        "security breach.",
        "What triggers a security incident notification to the customer, and who issues it?",
    ),
    "iso-soc-scope": (
        "Which security certifications do you currently hold, and what scope does each cover?",
        "List your security certifications and attestations, with the scope statement applying "
        "to each.",
        "Which security certifications cover the services proposed here, and what scope "
        "statement applies to each?",
    ),
    "knowledge-transfer": (
        "How is knowledge transferred to our own teams during and after the programme?",
        "What knowledge transfer and enablement do you provide so customer teams can operate "
        "the platform?",
        "What does knowledge transfer look like in practice, and how is team readiness "
        "measured before handover?",
    ),
    "landing-zone": (
        "What does your landing zone design look like, and how are guardrails applied and "
        "enforced?",
        "Set out the landing zone architecture you would build, including preventive and "
        "detective guardrails.",
        "How is the landing zone accounted and segmented, and which guardrails are preventive "
        "rather than detective?",
    ),
    "migration-tooling-selection": (
        "On what basis is migration tooling selected for a particular estate?",
        "How are migration tooling decisions made, and how is the selection justified to the "
        "customer?",
        "Which migration tooling would you propose for our estate, and why that selection?",
    ),
    "network-connectivity": (
        "What hybrid network connectivity is established during migration, and how is it secured?",
        "How is network connectivity maintained between on-premises and cloud environments "
        "through the migration?",
        "What network connectivity is required during the hybrid period, and how is it "
        "decommissioned afterwards?",
    ),
    "observability": (
        "What observability, monitoring and alerting do you put in place for migrated workloads?",
        "How is observability established for a workload once migrated, and who consumes it?",
        "What observability tooling is deployed for migrated workloads, and who holds the "
        "dashboards after handover?",
    ),
    "partnerships": (
        "Which cloud provider partnerships do you hold, and what competencies have you achieved?",
        "Set out your partnership status with each major cloud provider and any relevant "
        "competencies.",
        "At what tier do you partner with each cloud provider, and which competencies are "
        "currently held?",
    ),
    "performance-validation": (
        "How is workload performance validated after migration, and against what baseline?",
        "What performance validation and acceptance testing follows a migration?",
        "Which performance criteria must a migrated workload meet before it is accepted?",
    ),
    "quality-system": (
        "What quality management system governs delivery, and how is its maturity assessed?",
        "Set out your quality management system and the evidence of delivery maturity behind it.",
        "Which quality management standards do you certify against, and how is delivery "
        "maturity evidenced?",
    ),
    "raci-model": (
        "What RACI model applies across the migration programme?",
        "Set out responsibilities and accountabilities for the migration programme in a RACI.",
        "How are migration programme responsibilities split between your team and ours in a RACI?",
    ),
    "reference-experience": (
        "What comparable migration programmes have you delivered, and at what scale?",
        "Provide references for migration programmes of comparable size and complexity.",
        "Which migration programmes of comparable scope can you reference, and what were the "
        "outcomes?",
    ),
    "retention-continuity": (
        "How is key-person risk managed, and what ensures team continuity across the programme?",
        "What arrangements maintain continuity of the team, and how is dependence on key "
        "individuals reduced?",
        "What is your attrition rate on programmes of this length, and how is team continuity "
        "protected?",
    ),
    "sixr-strategy": (
        "How is the 6R disposition determined for each workload, and who signs it off?",
        "What process assigns a migration strategy to each workload, and how are 6R "
        "dispositions recorded?",
        "Which 6R disposition would each workload receive, and what evidence drives that "
        "migration strategy?",
    ),
    "subcontractor-model": (
        "Would any subcontractors be used on this programme, and how are they governed?",
        "What is your subcontractor model, and what controls and flow-down obligations apply "
        "to them?",
        "Which parts of the work would subcontractors deliver, and how is their performance "
        "governed?",
    ),
    "team-model": (
        "What team model would you deploy for a migration of this scale?",
        "Set out the team structure and staffing profile proposed for a migration of this size.",
        "Which roles make up the proposed team model, and how does it scale across migration "
        "waves?",
    ),
    "vulnerability-management": (
        "What is your vulnerability management and patching process across the estate?",
        "How are vulnerabilities identified, prioritised and patched, and to what timescales?",
        "How quickly are critical vulnerabilities patched, and what governs the patch "
        "management cycle?",
    ),
    "warranty-process": (
        "What warranty applies after migration, and for how long does it run?",
        "Set out your post-migration warranty arrangements and how defects are handled within "
        "them.",
        "What does the post-migration warranty cover, and how are defects raised within it?",
    ),
    "wave-planning": (
        "How are workloads sequenced into migration waves, and what drives the ordering?",
        "What determines the composition and order of migration waves across the portfolio?",
        "Which criteria decide the migration wave a given workload is placed in?",
    ),
}
