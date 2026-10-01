# Dataset audit: alpie/indian-law

- File: `data/raw/Alpie-core_core_indian_law.json` (258,243,053 bytes, sha256 `bd7586b619d62a03…`)
- Rows: **47,789**
- Columns: prompt, complex_cot, response

## Missing / empty
| column | null | empty/whitespace |
|---|---|---|
| prompt | 0 | 0 |
| complex_cot | 0 | 0 |
| response | 0 | 0 |

## Length (words)
| column | mean | p5 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| prompt | 18.66 | 9 | 17 | 33 | 42 | 135 |
| complex_cot | 496.99 | 154 | 417 | 1085 | 1960 | 27062 |
| response | 282.39 | 59 | 263 | 590 | 834 | 21788 |

## Duplicates
- exact_full_row: 23
- exact_prompt: 11841
- normalized_prompt: 11885
- normalized_prompt_and_response: 114
- exact_response: 203
- prompt_groups_with_multiple_rows: 9784
- largest_prompt_group: 86
- prompt_groups_with_conflicting_responses: 9704

Most repeated prompts:
- (86×) can state legislatures introduce additional measures to support and empower scheduled castes scheduled tribes and women chairpersons
- (86×) how do these provisions align with broader national goals or policies related to social justice equality and inclusion
- (85×) what are the potential long term implications of implementing these reservation requirements in panchayats
- (84×) are there any ongoing debates or discussions around the effectiveness and fairness of these reservation requirements for panchayat positions
- (43×) what are the changes brought about by the constitution first amendment act 1951 s 13 in terms of office tenure for the person who was auditor general 
- (28×) chapter number 16 under offences affecting the human body
- (24×) how does leave of absence work for the comptroller and auditor general of india under this provision
- (22×) how does the comptroller and auditor general s term of office change under this provision
- (22×) what are the changes brought about by the constitution first amendment act 1951 s 13 in terms of salaries and pensions for the person who was auditor 
- (20×) how does the continuation of office work for the person who was auditor general immediately before the commencement of this constitution according to 

## Near-duplicate prompts (TF-IDF cosine ≥ 0.9)
- Rows whose nearest neighbour is above threshold: 22559 (47.21%)
- NN similarity quantiles: {'p50': 0.824, 'p75': 1.0, 'p90': 1.0, 'p95': 1.0, 'p99': 1.0}
  - 0.908: "What is the limitation period for filing a suit for malicious prosecution?" ↔ "What is the limitation period for filing a suit for compensation for malicious prosecution?"
  - 0.944: "Discuss the role of the Intellectual Property Appellate Board (IPAB) in India. What types of cases does it handle?" ↔ "What is the role of the Intellectual Property Appellate Board (IPAB) in India, and what types of cases does it handle?"
  - 0.985: "What is the role of the Registrar of Firms in the registration and regulation of partnership firms in India?" ↔ "Explain the role of the Registrar of Firms in the registration and regulation of partnership firms in India."
  - 0.924: "Under the Motor Vehicles Act, 1988, what is the minimum third-party insurance cover required for a private car in India?" ↔ "Under the Motor Vehicles Act, what is the minimum third-party insurance cover required for a private car in India?"
  - 0.972: "What are the legal provisions for the regulation of noise pollution in India under the Noise Pollution (Regulation and C" ↔ "What are the provisions for the regulation of noise pollution in India under the Noise Pollution (Regulation and Control"

## Task format
- prompts_with_mcq_pattern: 382
- prompts_ending_with_question_mark: 38665
- prompts_yes_no_style_start: 4142
- responses_starting_yes_or_no: 2851
- prompts_context_dependent: 667
- first words: {'what': 20269, 'how': 6039, 'explain': 4542, 'discuss': 3221, 'can': 2617, 'a': 1431, 'when': 1091, 'are': 954, 'which': 817, 'who': 783, 'in': 532, 'article': 465, 'under': 458, 'analyze': 445, 'is': 318, 'describe': 226, 'ramesh': 198, 'mr.': 164, 'advocate': 163, 'the': 157}

Context-dependent examples:
- How are 'Shall' and 'May' interpreted in statutes, and what factors would a court consider in deciding whether 'Shall' is mandatory or directory in a given context?
- The Wildlife Protection Act, 1972 prohibits hunting of endangered species. A person claims they hunted an endangered species for self-defense. Discuss the defense of necessity and how the Act should b
- The government wants to acquire land in Kerala for a solar power project. Discuss the issues related to environmental clearance and land acquisition in this context.
- In a case involving intellectual property rights in Mumbai, the plaintiff relies on documentary evidence to prove ownership. Discuss the admissibility and relevance of secondary evidence in this conte
- What are India's rights and obligations concerning the passage of foreign warships through its territorial waters?
- A agrees to supply B with 'necessary articles' suitable to B's condition in life. What constitutes 'necessary articles' in this context?
- During a property dispute in Mumbai, a 50-year-old land deed is presented in court. What is the relevance of Section 90 of the Evidence Act concerning this document?
- A document is produced in court claiming that it is the last will and testament of Mr. Sharma from Jaipur. What steps must be taken to prove the genuineness of this document under the Indian Evidence 

## Reasoning field
- field: complex_cot
- empty: 0
- contains_think_or_answer_markers: 11803
- contains_first_80_chars_of_response: 511
- note: complex_cot is gold reasoning; it must never be given to the evaluated model at inference.

## Domain / jurisdiction signals
- definition: regex heuristics over prompt+response; see INDIAN_MARKERS / FOREIGN_MARKERS in code
- rows_with_indian_marker: 28024
- rows_with_foreign_marker: 2745
- rows_with_foreign_but_no_indian_marker: 1629
- rows_with_neither: 18136
- rows_with_devanagari: 58
- non_ascii_char_fraction_prompt: 7e-05

Foreign-only examples:
- Jane, a Christian, alleges adultery by her husband, Peter. What is the standard of proof required to establish adultery as a ground for divorce under the Divorce Act, 1869?
- What is the difference between a simple mortgage and an usufructuary mortgage under the Transfer of Property Act, 1882? Provide examples from real estate transactions in Chennai.
- A trust is created in Tamil Nadu for the maintenance of a temple. What are the specific provisions of the Hindu Religious and Charitable Endowments Act that might apply?
- How does the RFCTLARR Act address the issue of adverse possession of land?
- Explain the duties of a manufacturer under the Consumer Protection Act, 2019.
- A statute provides for 'compensation' to victims of a certain crime. How is the amount of compensation determined?
- What is the importance of showing 'past conduct' between parties applying for specific performance under the Specific Relief Act?
- Discuss the concept of 'Input Tax Credit' (ITC) under GST, and how it can be utilized to offset output tax liability.

No-marker examples:
- Explain the concept of 'workman' as defined under the Industrial Disputes Act, 1947, and how it applies to a software engineer employed on a contractual basis in Bangalore.
- What is the role of 'Judicial Precedent' in the interpretation of statutes, and how does 'Stare Decisis' affect the process?
- Explain the concept of 'severability' in arbitration agreements and its importance.
- Discuss the role of a 'protector' in a trust, and the powers that can be conferred upon them.
- How does the Arbitration and Conciliation Act, 1996, deal with the issue of multi-party arbitration?
- Explain the 'Order XXI Rule 32' of CPC related to Execution of decree for specific performance. Illustrate with a case where the judgment debtor fails to execute the sale deed, despite a decree for sp
- What are the restrictions on the purchase by a company, 'Trivandrum Spices Ltd.', of its own shares?
- Section 25 of the Limitation Act deals with the acquisition of easement rights. How is the period of 'peaceable enjoyment' calculated for acquiring an easement?

## Legal categories
No category field exists. Statute mentions in prompts (proxy):
- explicit_category_field: None
- prompts_mentioning_a_named_act_with_year: 8078
- prompts_mentioning_article: 3263
- prompts_mentioning_section: 2047
- distinct_acts_in_prompts: 512
  - IT Act 2000: 505
  - Property Act 1882: 469
  - Conciliation Act 1996: 468
  - Consumer Protection Act 2019: 379
  - Limitation Act 1963: 335
  - Indian Evidence Act 1872: 312
  - Companies Act 2013: 310
  - Negotiable Instruments Act 1881: 293
  - Motor Vehicles Act 1988: 240
  - Seventh Amendment) Act 1956: 193
  - Income Tax Act 1961: 190
  - Indian Contract Act 1872: 184
  - Industrial Disputes Act 1947: 156
  - Indian Partnership Act 1932: 115
  - Specific Relief Act 1963: 112
  - Amendment) Act 1978: 109
  - Land Acquisition Act 2013: 107
  - Amendment) Act 1976: 102
  - First Amendment) Act 1951: 100
  - Resettlement Act 2013: 80
  - Information Technology Act 2000: 79
  - Amendment) Act 1975: 76
  - Indian Trusts Act 1882: 75
  - Land Acquisition Act 1894: 68
  - Reorganisation) Act 1971: 68
  - Amendment) Act 2003: 64
  - Hindu Marriage Act 1955: 45
  - Amendment) Act 1992: 43
  - Diu Reorganisation Act 1987: 43
  - Factories Act 1948: 41

## Templating signal
- {'prompts_mentioning_major_city': 2457, 'city_counts': {'Delhi': 521, 'Mumbai': 408, 'Chennai': 301, 'Kolkata': 292, 'Bangalore': 274, 'Jaipur': 174, 'Hyderabad': 164, 'Lucknow': 121, 'Ahmedabad': 95, 'Pune': 77, 'Guwahati': 51, 'Chandigarh': 40, 'Kochi': 39, 'Bhopal': 39, 'Bengaluru': 6}}

## Split / labels
- predefined_split: none (single file, no split column)
- label_columns: []
- difficulty_labels: none
- task_type: free-form question -> long-form answer generation (with reasoning trace)
