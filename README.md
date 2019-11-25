# Essay grading for Orange

This is an essay grading add-on for [Orange data mining](https://orange.biolab.si/).

## Installation

- [Download Orange](https://orange.biolab.si/download/)

- Clone this repository


- Open Anaconda prompt, navigate to cloned repository, activate Orange environment and download additional vocabulary files:

        cd essay-grading
        activate "C:\Program Files\Orange"
        python -m spacy download en_vectors_web_lg

- Open Orange
    - Click Options > Add-ons
    - Install Orange3-Text add-on
    
- DONE!
    - Code changes will take effect immediately or after restarting Orange (if open).

## Usage

- Run Orange
- All essay grading widgets are bundled in "Essay grading"
- Open provided Orange test file for example usage


pip install spacy

python -m spacy download en_vectors_web_lg
