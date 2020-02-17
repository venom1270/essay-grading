if __name__ == "__main__":
    from orangecontrib.essaygrading.utils.OntologyUtils import run_semantic_consistency_check

    essays = [["Lisa is a boy.", "Lisa is a girl."],
              ["Tennis is a fast sport.", "Lisa doesn't like fast sports.", "She likes tennis.", "Football is a fast sport.", "Lisa likes football.", ]]

    run_semantic_consistency_check(essays, use_coref=True, explain=True)
