# Essay grading for Orange

This is an essay grading add-on for [Orange data mining](https://orange.biolab.si/).

## Installation

- [Download and install Orange](https://orange.biolab.si/download/)

- Clone this repository

- Open Anaconda prompt (as Admin), navigate to cloned repository, activate Orange environment, install essay grading, and download additional vocabulary files:

        cd essay-grading
        activate "C:\Program Files\Orange"
        pip install -e .
        python -m spacy download en_vectors_web_lg

- Open Orange
    - Click Options > Add-ons
    - Install Orange3-Text add-on
    
- DONE!
    - Code changes will take effect immediately or after restarting Orange (if open).

## Usage

- Run Orange
- All essay grading widgets are bundled in "Essay grading"
- [Open provided Orange test file for example usage](https://github.com/venom1270/essay-grading-util)
     - example.ows is an Orange example model
     - set2A.tsv is the file you should load in Corpus widget


## Remarks

This is an implementation of an essay grading system described in [PhD thesis](http://eprints.fri.uni-lj.si/4133/1/63120364-KAJA_ZUPANC-Semanti%C4%8Dno_usmerjeno_avtomatsko_ocenjevanje_esejev.pdf) by Kaja Zupanc.

AGE and AGE+ systems are fully functional. SAGE is functional on a conceptual level, but needs a few more tweaks and optimizations to be usable in real world examples.