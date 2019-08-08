from setuptools import setup

setup(name="Demo",
      packages=["orangedemo"],
      #package_data={"orangedemo": ["icons/*.svg", "dale_chall_word_list.txt"]},
      include_package_data=True,
      classifiers=["Example :: Invalid"],
      # Declare orangedemo package to contain widgets for the "Demo" category
      entry_points={"orange.widgets": "Demo = orangedemo"},
      )
