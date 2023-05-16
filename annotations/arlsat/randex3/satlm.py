### write python code to answer the question
"""
In a repair facility there are exactly six technicians: Stacy, Urma, Wim, Xena, Yolanda, and Zane. Each technician repairs machines of at least one of the following three types—radios, televisions, and VCRs—and no other types. The following conditions apply: Xena and exactly three other technicians repair radios. Yolanda repairs both televisions and VCRs. Stacy does not repair any type of machine that Yolanda repairs. Zane repairs more types of machines than Yolanda repairs. Wim does not repair any type of machine that Stacy repairs. Urma repairs exactly two types of machines.
Question: Which one of the following pairs of technicians could repair all and only the same types of machines as each other?
Choices:
(A) Stacy and Urma
(B) Urma and Yolanda
(C) Urma and Xena
(D) Wim and Xena
(E) Xena and Yolanda
"""
# declare variables
technicians = EnumSort([Stacy, Urma, Wim, Xena, Yolanda, Zane])
machines = EnumSort([radios, televisions, VCRs])
repairs = Function([technicians, machines] -> [bool])

# constraints
# each technician repairs machines of at least one of the following three types
ForAll([t:technicians], Count([m:machines], repairs(t, m)) >= 1)

# Xena and exactly three other technicians repair radios
And(repairs(Xena, radios), Count([t:technicians], And(t != Xena, repairs(t, radios))) == 3)

# Yolanda repairs both televisions and VCRs
And(repairs(Yolanda, televisions), repairs(Yolanda, VCRs))

# Stacy does not repair any type of machine that Yolanda repairs
ForAll([m:machines], Implies(repairs(Yolanda, m), Not(repairs(Stacy, m))))

# Zane repairs more types of machines than Yolanda repairs
Count([m:machines], repairs(Zane, m)) > Count([m:machines], repairs(Yolanda, m))

# Wim does not repair any type of machine that Stacy repairs
ForAll([m:machines], Implies(repairs(Stacy, m), Not(repairs(Wim, m))))

# Urma repairs exactly two types of machines
Count([m:machines], repairs(Urma, m)) == 2

# Which one of the following pairs of technicians could repair all and only the same types of machines as each other?
# we check whether the options can possibly be true
# (A)
is_sat(ForAll([m:machines], repairs(Stacy, m) == repairs(Urma, m)))
# (B)
is_sat(ForAll([m:machines], repairs(Urma, m) == repairs(Yolanda, m)))
# (C)
is_sat(ForAll([m:machines], repairs(Urma, m) == repairs(Xena, m)))
# (D)
is_sat(ForAll([m:machines], repairs(Wim, m) == repairs(Xena, m)))
# (E)
is_sat(ForAll([m:machines], repairs(Xena, m) == repairs(Yolanda, m)))