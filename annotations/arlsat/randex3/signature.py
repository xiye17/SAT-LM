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
repairs = Function(technicians, machines, bool)

# Question: Which one of the following pairs of technicians could repair all and only the same types of machines as each other?
# we check whether the options can possibly be true
print(check_sat())