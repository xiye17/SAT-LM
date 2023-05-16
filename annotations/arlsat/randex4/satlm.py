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
speaks = Function([experts] -> [languages])
order = Function([experts] -> [int])
ForAll([e:experts], And(1 <= order(e), order(e) <= 5))

# constraints
# The five presentations are given consecutively
Distinct([e:experts], order(e))

# Exactly two of the presentations are in the same language as each other
Count([l:languages], Count([e:experts], speaks(e) == l) == 2) == 1

# The statistician gives the second presentation in Hindi
And(order(statistician) == 2, speaks(statistician) == Hindi)

# The lawyer gives the fourth presentation in either Mandarin or French
And(order(lawyer) == 4, Or(speaks(lawyer) == Mandarin, speaks(lawyer) == French))

# The oceanographer presents in either French or Japanese; the same is true of the physicist
And(Or(speaks(oceanographer) == French, speaks(oceanographer) == Japanese), Or(speaks(physicist) == French, speaks(physicist) == Japanese))

# The first presentation and the last presentation are in Japanese
And(ForAll([e:experts], Implies(order(e) == 1, speaks(e) == Japanese)), ForAll([e:experts], Implies(order(e) == 5, speaks(e) == Japanese)))

# Which one of the following could be the order in which the experts give their presentations, from first to last?
# we check whether the options can possibly be true
# (A)
is_sat(And(order(physicist) == 1, order(statistician) == 2, order(lawyer) == 3, order(naturalist) == 4, order(oceanographer) == 5))
# (B)
is_sat(And(order(physicist) == 1, order(naturalist) == 2, order(oceanographer) == 3, order(lawyer) == 4, order(statistician) == 5))
# (C)
is_sat(And(order(oceanographer) == 1, order(statistician) == 2, order(naturalist) == 3, order(lawyer) == 4, order(physicist) == 5))
# (D)
is_sat(And(order(oceanographer) == 1, order(statistician) == 2, order(lawyer) == 3, order(naturalist) == 4, order(physicist) == 5))