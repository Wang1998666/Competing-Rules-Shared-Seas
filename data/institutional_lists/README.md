# Institutional lists

The two country sets below drive the governance attribution classifier
(module `12_governance_classification.py`, with the EU/EEA set also used by
module `03c_h3_policy_coverage.py`). They are shipped here so the
classification is auditable without reading code, and they are reproduced
verbatim from the hard-coded sets in `code/calc_v5/08_bri_analysis.py` and
`code/calc_v5/03c_h3_policy_coverage.py`.

## eu_eea_countries.csv — 30 codes

ISO2 codes of the 27 EU member states plus the 3 EEA (EFTA) states
(Iceland, Liechtenstein, Norway), as used by the EU ETS maritime rules.

Sources (see the manuscript’s Open Research statement for the citations):

- EU member states: <https://european-union.europa.eu/principles-countries-history/country-profiles_en>
- EEA states: <https://www.efta.int/eea>
- The ETS port jurisdiction (including the French outermost regions) is defined
  by Directive (EU) 2023/959 amending Directive 2003/87/EC:
  <https://eur-lex.europa.eu/eli/dir/2023/959/oj>

Note: French outermost regions are included in the ETS jurisdictional scope by
the directive but are deliberately **excluded** from the corridor-level rule
definition in the classifier (they would misclassify open-ocean corridors);
this is documented in `03c_h3_policy_coverage.py`.

## bri_maritime_partners.csv — 59 codes

ISO2 codes of the 59 Belt and Road maritime partner countries used to label
corridors as BRI when an endpoint country is a member.

Sources:

- World Bank (2019), *Belt and Road Economics: Opportunities and Risks of
  Transport Corridors*: <https://www.worldbank.org/en/topic/regional-integration/publication/belt-and-road-economics-opportunities-and-risks-of-transport-corridors>
- Membership follows the official BRI partner-country list as of 2024; the
  Maritime Silk Road subset (ports reachable by sea) is what enters the
  classifier. Ports of land-BRI members (Central Asia via the Caspian, Russia
  via the Northern Sea Route) are included because the dataset carries their
  port calls.

Both files are CC0 (public domain) reproductions of official lists.
