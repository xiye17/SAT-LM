### write python code to answer the question
"""
Workers at a water treatment plant open eight valves—G, H, I, K, L, N, O, and P—to flush out a system of pipes that needs emergency repairs. To maximize safety and efficiency, each valve is opened exactly once, and no two valves are opened at the same time. The valves are opened in accordance with the following conditions: Both K and P are opened before H. O is opened before L but after H. L is opened after G. N is opened before H. I is opened after K.
Question: Each of the following could be the fifth valve opened EXCEPT:
Choices:
(A) H
(B) I
(C) K
(D) N
(E) O
"""
# declare variables
valves = EnumSort([G, H, I, K, L, N, O, P])
opened = Function(valves, int)
Distinct([v:valves], opened(v))
ForAll([v:valves], And(1 <= opened(v), opened(v) <= 8))

# Question: Each of the following could be the fifth valve opened EXCEPT:
# we check whether the options can possibly be true, and find the exception
print(exception(check_sat()))