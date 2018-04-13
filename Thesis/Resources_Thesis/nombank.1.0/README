1. Description: 

NomBank is an annotation project at New York University that is
related to the PropBank project at the University of Pennsylvania.
Our goal is to mark the sets of arguments that cooccur with nouns in
the PropBank Corpus (the Wall Street Journal Corpus of the Penn Treebank),
just as PropBank records such information for verbs.

2. Release Information:

This is the nombank.1.0 release relased December 17, 2007. It includes
all the markable nouns in the Wall Street Journal Corpus Penn Treebank
II/PropBank corpus. During the final stages, we completed our analysis
of the remaining NOMLEX classes (we had saved the rarer classes for
the final stages) and have performed extensive quality control
measures based on syntactic patterns and lexical information also
updated all dictionaries related to this effort and all
specifications. For previous releases, we looked at about 21,000
instances tagged as likely errors. For this release, we looked at an
additional 13,000 instances.

This release includes a total of 114,576 propositions, including 42
instances of nouns that license 2 propositions.  These instances
include 4704 different words. Including unmarked instances of these
words, the NomBank team have approximately 202,965 noun instances,
leaving 88,347 nouns unmarked because they contained no arguments.
An additional 35,000 nouns were ignored because it was
determined that these nouns never occurred with any arguments.

This release also includes substantial updates of the documentation to
include more details about NOMBANK classes, especially about NOMBANK
entries derived from adjectives adn adverbs. An additional document
entitled "Those Other NomBank Dictionaries" is also included in order
to provide clearer descriptions of the supplemental dictionaries
included with NomBank.

3. Contents:

README - this file

README-dictionaries - description of format of dictionary files 

README-nombank-proposition-structure - describes format of NomBank 
				       propositions in proposition files

nombank.1.0 - NomBank Propositions

nombank.1.0.words - a list of all the words covered by NomBank

adjudicated -- The subcorpus of nombank that has been annotated by at
	       least two annotators (usually 4) and have been
	       adjudicated by an additional annotator or
	       supervisor. It is expected that instances of these
	       words will have the best accuracy and coverage.

Dictionary files: NOMLEX-plus.1.0 (includes some information automatically
			           extracted from annotation)
		  NOMLEX-plus-training.1.0 (includes automatically
				            extracted info from
				            sections 2 to 21 only for
				            those who divide the
				            treebank into training,
				            test and development
				            corpora)
                  NOMLEX-plus-clean.1.0 (includes no automatically extracted 
					 corpus information)
		  ADJADV.1.0
		  NOMADV.1.0
                  nombank-morph.dict.1.0
		  frames/*.xml
		  nombank-dict.1.0 (a lispified version of all frame files which includes
				    :GENDESCR features for roles see the document
				    entitled "Those Other NomBank Dictionaries" included
				    in this release)

Addresses-of-Multiple-Propositions - A list of Addresses, each of
				     which corresponds to two
				     propositions rather than just
				     one.

4. Related Files that require LDC licences:

COMNOM.1.0 -- a version of COMLEX Syntax that has been automatically
	      updated based on NOMLEX-plus. Their are three versions
	      of COMNOM, each derived from the version of NOMLEX-plus
	      that shares its suffix: COMNOM.1.0, COMNOM.1.0-tr and
	      COMNOM.1.0-clean. Users require a license for COMLEX Syntax
	      to obtain these.

nombank.1.0.print -- a printout of nombank.1.0 in human readable
		  form. Users require a license to the WSJ corpus in
		  order to obtain a copy of this document. For
		  example, consider the following line from
		  nombank.1.0:

wsj/00/wsj_0044.mrg 22 5 ability 01 4:0-ARG0 5:0-rel 6:2-ARG1-PRD

This is printed out as the following:

---------------------------------------------------------

Example: The bonus depended on her ---> ABILITY <--- to produce higher
student-test scores .

Address: wsj/00/wsj_0044.mrg 22 5

Example Sense: 1
Parts of the Proposition:
(ARG :N "0" :STRINGS ("her"))
(REL :STRINGS ("ability"))
(ARG :N "1" :F "PRD" :STRINGS ("to produce higher student-test scores"))

---------------------------------------------------------

5. Documentation can be found both separately on the website and in the
DOCS subdirectory. It consists of:

nombank-specs-2007.pdf -- An updated manual for nombank, the frame files
and the morphology dictionary (nombank-morph.dict).

those-other-nombank-dictionaries.pdf -- describes the other
dictionaries related to the NomBank project. To fully use NOMLEX-PLUS,
it is recommended that user consult the original NOMLEX manual for the
regularized version of NOMLEX (available at
http://nlp.cs.nyu.edu/nomlex/index.html). To fully use COMNOM, it is
suggested that the user consult the original COMLEX Syntax manual
(available from http://nlp.cs.nyu.edu/comlex/).

For further discussion of these dictionaries, please read "The
Cross-Breeding of Dictionaries" which we presented at LREC-2004. A
printable copy of our paper from the proceedings is on the NomBank
website. 

6. Website: http://nlp.cs.nyu.edu/meyers/NomBank.html
