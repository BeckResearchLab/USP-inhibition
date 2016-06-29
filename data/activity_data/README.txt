PC-AssayDescription ::= {
  aid {
    id 743255,
    version 1
  },
  aid-source db {
    name "NCGC",
    source-id str "usp1-uaf1-p1"
  },
  name "Inhibitors of USP1/UAF1: Primary Screen",
  description {
    "Deubiquitinating enzymes (DUBs) are a class of enzymes that can cleave
 isopeptide bond formed between the C-terminal carboxylate of ubiquitin and a
 lysine side-chain amino group on the target protein. Among them, ubiquitin
 specific proteases (USPs) constitute the largest DUB family. Human USPs are
 emerging as promising targets for pharmacological intervention because of
 their connection to a number of human diseases, including prostate, colon and
 breast cancer (1, 2), pediatric acute lymphoblastic leukemia (3), and
 familial cylindromatosis (4). The advantage of inhibiting USPs lies in the
 potential specificity of therapeutic intervention that can lead to better
 efficacy and reduce nonspecific side effects.",
    "",
    "In a collaboration between the University of Delaware and the NIH
 Chemical Genomics Center, a high-throughput screen assay was developed to
 screen for USP1/UAF1 inhibitors. This miniaturized assay has a fluorescent
 read-out and is used to screen the NIH Molecular Libraries Small Molecule
 Repository (MLSMR) in order to identify a small molecule that inhibits USP1.",
    "",
    "NIH Chemical Genomics Center [NCGC]",
    "NIH Molecular Libraries Probe Centers Network [MLPCN]",
    "",
    "MLPCN Grant: DA030552",
    "Assay Submitter (PI): Zhihao Zhuang, University of Delaware",
    "",
    "1. Priolo, C., et al. (2006) The isopeptidase USP2a protects human
 prostate cancer from apoptosis. Cancer Res 66, 8625-8632",
    "",
    "2. Popov, N., et al. (2007) The ubiquitin-specific protease USP28 is
 required for MYC stability. Nat Cell Biol 9, 765-774",
    "",
    "3. De Pitta, C., et al. (2005) A leukemia-enriched cDNA microarray
 platform identifies new transcripts with relevance to the biology of
 pediatric acute lymphoblastic leukemia. Haematologica 90, 890-898",
    "",
    "4. Kovalenko, A., et al. (2003) The tumour suppressor CYLD negatively
 regulates NF-kappaB signalling by deubiquitination. Nature 424, 801-805"
  },
  protocol {
    "3 microliters of reagents (buffer in columns 3 and 4 as negative control
 and 1nM USP1/UAF1 complex in columns 1, 2, and 5-48) are dispensed into
 Greiner black solid bottom 1536-well assay plates. Compounds are then
 transferred via Kalypsys pin tool equipped with 1536-pin array (10nl slotted
 pins, V&P Scientific, San Diego, CA). Following an incubation step of 15 min
 at room temperature, 1ul of Ub-Rho substrate (150nM final concentration) is
 added to initiate the reaction. The plates are immediately centrifuged at
 1000 rpm for 15 seconds, and subsequently transferred to a ViewLux
 high-throughput CCD imager (PerkinElmer) wherein kinetic measurements of
 fluorescence are acquired using 480 nm excitation/540 nm emission filter set
 (6 reads every 60 seconds, see Table 1). All reagents are diluted in an assay
 buffer consisting of 50mM HEPES (pH 7.8), 0.5mM EDTA, 0.1 mg/ml BSA, 1mM
 TCEP, and 0.01% Tween-20."
  },
  comment {
    "Compound Ranking:",
    "",
    "1. Compounds are first classified as having full titration curves,
 partial modulation, partial curve (weaker actives), single point activity (at
 highest concentration only), or inactive. See data field ""Curve Description
"". For this assay, apparent inhibitors are ranked higher than compounds that
 showed apparent activation.",
    "2. For all inactive compounds, PUBCHEM_ACTIVITY_SCORE is 0. For all
 active compounds, a score range was given for each curve class type given
 above.  Active compounds have PUBCHEM_ACTIVITY_SCORE between 40 and 100. 
 Inconclusive compounds have PUBCHEM_ACTIVITY_SCORE between 1 and 39. 
 Fit_LogAC50 was used for determining relative score and was scaled to each
 curve class' score range."
  },
  xref {
    {
      xref aid 504878,
      comment "Summary AID"
    },
    {
      xref pmid 12917691
    },
    {
      xref pmid 15996926
    },
    {
      xref pmid 16951176
    },
    {
      xref pmid 17558397
    },
    {
      xref gene 7398
    },
    {
      xref mim 603478
    },
    {
      xref taxonomy 9606
    },
    {
      xref dburl "http://www.ncgc.nih.gov"
    }
  },
  results {
    {
      tid 1,
      name "Phenotype",
      description {
        "Indicates type of activity observed: inhibitor, activator,
 fluorescent, cytotoxic, inactive, or inconclusive."
      },
      type string,
      unit none
    },
    {
      tid 2,
      name "Potency",
      description {
        "Concentration at which compound exhibits half-maximal efficacy, AC50.
 Extrapolated AC50s also include the highest efficacy observed and the
 concentration of compound at which it was observed."
      },
      type float,
      unit um,
      ac TRUE
    },
    {
      tid 3,
      name "Efficacy",
      description {
        "Maximal efficacy of compound, reported as a percentage of control.
 These values are estimated based on fits of the Hill equation to the
 dose-response curves."
      },
      type float,
      unit percent
    },
    {
      tid 4,
      name "Analysis Comment",
      description {
        "Annotation/notes on a particular compound's data or its analysis."
      },
      type string,
      unit none
    },
    {
      tid 5,
      name "Activity_Score",
      description {
        "Activity score."
      },
      type int,
      unit none
    },
    {
      tid 6,
      name "Curve_Description",
      description {
        "A description of dose-response curve quality. A complete curve has
 two observed asymptotes; a partial curve may not have attained its second
 asymptote at the highest concentration tested. High efficacy curves exhibit
 efficacy greater than 80% of control. Partial efficacies are statistically
 significant, but below 80% of control."
      },
      type string,
      unit none
    },
    {
      tid 7,
      name "Fit_LogAC50",
      description {
        "The logarithm of the AC50 from a fit of the data to the Hill equation
 (calculated based on Molar Units)."
      },
      type float,
      unit none
    },
    {
      tid 8,
      name "Fit_HillSlope",
      description {
        "The Hill slope from a fit of the data to the Hill equation."
      },
      type float,
      unit none
    },
    {
      tid 9,
      name "Fit_R2",
      description {
        "R^2 fit value of the curve. Closer to 1.0 equates to better Hill
 equation fit."
      },
      type float,
      unit none
    },
    {
      tid 10,
      name "Fit_InfiniteActivity",
      description {
        "The asymptotic efficacy from a fit of the data to the Hill equation."
      },
      type float,
      unit percent
    },
    {
      tid 11,
      name "Fit_ZeroActivity",
      description {
        "Efficacy at zero concentration of compound from a fit of the data to
 the Hill equation."
      },
      type float,
      unit percent
    },
    {
      tid 12,
      name "Fit_CurveClass",
      description {
        "Numerical encoding of curve description for the fitted Hill equation."
      },
      type float,
      unit none
    },
    {
      tid 13,
      name "Excluded_Points",
      description {
        "Which dose-response titration points were excluded from analysis
 based on outlier analysis. Each number represents whether a titration point
 was (1) or was not (0) excluded, for the titration series going from smallest
 to highest compound concentrations."
      },
      type string,
      unit none
    },
    {
      tid 14,
      name "Max_Response",
      description {
        "Maximum activity observed for compound (usually at highest
 concentration tested)."
      },
      type float,
      unit percent
    },
    {
      tid 15,
      name "Activity at 0.457 uM",
      description {
        "% Activity at given concentration."
      },
      type float,
      unit percent,
      tc {
        concentration { 456999987363815, 10, -15 },
        unit um,
        dr-id 1
      }
    },
    {
      tid 16,
      name "Activity at 2.290 uM",
      description {
        "% Activity at given concentration."
      },
      type float,
      unit percent,
      tc {
        concentration { 228999996185303, 10, -14 },
        unit um,
        dr-id 1
      }
    },
    {
      tid 17,
      name "Activity at 11.40 uM",
      description {
        "% Activity at given concentration."
      },
      type float,
      unit percent,
      tc {
        concentration { 113999996185303, 10, -13 },
        unit um,
        dr-id 1
      }
    },
    {
      tid 18,
      name "Activity at 57.10 uM",
      description {
        "% Activity at given concentration."
      },
      type float,
      unit percent,
      tc {
        concentration { 570999984741211, 10, -13 },
        unit um,
        dr-id 1
      }
    },
    {
      tid 19,
      name "Compound QC",
      description {
        "NCGC designation for data stage: 'qHTS', 'qHTS Verification', 
'Secondary Profiling'"
      },
      type string,
      unit none
    }
  },
  revision 1,
  target {
    {
      name "USP1 protein [Homo sapiens]",
      mol-id 118600387,
      molecule-type protein,
      organism {
        org {
          taxname "Homo sapiens",
          common "human",
          db {
            {
              db "taxon",
              tag id 9606
            }
          }
        }
      }
    }
  },
  activity-outcome-method confirmatory,
  dr {
    {
      id 1,
      descr "CR Plot label 1",
      dn "Concentration",
      rn "Response",
      type experimental
    }
  },
  grant-number {
    "DA030552"
  },
  project-category mlpcn
}
