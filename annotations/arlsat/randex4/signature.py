### write python code to answer the question
"""
Each of five experts—a lawyer, a naturalist, an oceanographer, a physicist, and a statistician—individually gives exactly one presentation at a conference. The five presentations are given consecutively. Each presentation is in exactly one of the four following languages: French, Hindi, Japanese, or Mandarin. Each expert speaks exactly one of the languages. The following conditions must hold: Exactly two of the presentations are in the same language as each other. The statistician gives the second presentation in Hindi. The lawyer gives the fourth presentation in either Mandarin or French. The oceanographer presents in either French or Japanese; the same is true of the physicist. The first presentation and the last presentation are in Japanese.
Question: Which one of the following could be the order in which the experts give their presentations, from first to last?
Choices:
(A) the physicist, the statistician, the lawyer, the naturalist, the oceanographer
(B) the physicist, the naturalist, the oceanographer, the lawyer, the statistician
(C) the oceanographer, the statistician, the naturalist, the lawyer, the physicist
(D) the oceanographer, the statistician, the lawyer, the naturalist, the physicist
"""
# declare variables
experts = EnumSort([lawyer, naturalist, oceanographer, physicist, statistician])
languages = EnumSort([French, Hindi, Japanese, Mandarin])
speaks = Function(experts, languages)
order = Function(experts, int)
Distinct([e:experts], order(e))
ForAll([e:experts], And(1 <= order(e), order(e) <= 5))

# Question: Which one of the following could be the order in which the experts give their presentations, from first to last?
# we check whether the options can possibly be true
print(check_sat())