import scipy.odr
import numpy as np
from pyscipopt import Model, Branchrule, SCIP_RESULT, quicksum, SCIP_PARAMSETTING

# def get_constraint_matrix(model, tolerance=1e-8):
#     constraints = model.getConss()
#     variables = model.getVars()
#     var_names = [var.name for var in variables]
#     print("var_names:", var_names)
#     constraint_matrix = []
#     vec_b = []
#     for cons in constraints:
#         row = [0] * len(variables)
#         conshdlrname = cons.getConshdlrName()
#         if conshdlrname == "linear":
#             linvals = model.getValsLinear(cons)
#             if linvals is None:
#                 print("Error: Linear constraint has wrong number of coefficients")
#             else:
#                 for var_name, coeff in linvals.items():
#                     short_var_name = var_name.split("_", 1)[-1]
#                     if abs(coeff) > tolerance:
#                         row[var_names.index(short_var_name)] = coeff
#
#         elif conshdlrname == "setppc":
#             for i in range(len(variables)):
#                 row[i] = 1
#         elif conshdlrname == "logicor":
#             for i in range(len(variables)):
#                 row[i] = 1
#         elif conshdlrname == "knapsack":
#             weights = model.getWeightsKnapsack(cons)
#             for i in range(len(variables)):
#                 row[i] = weights[i]
#         elif conshdlrname == "varbound":
#             assert len(variables) == 2
#             row[0] = 1
#             row[1] = model.getVbdcoefVarbound(cons)
#         elif conshdlrname == "SOS1":
#             weights = model.getWeightsSOS1(cons)
#             for i in range(len(variables)):
#                 row[i] = weights[i]
#         elif conshdlrname == "SOS2":
#             weights = model.getWeightsSOS2(cons)
#             for i in range(len(variables)):
#                 row[i] = weights[i]
#         else:
#             print("Error: Constraint handler not supported")
#         b = model.getLhs(cons)
#         if model.getLhs(cons) == -1e+20:
#             row = [-1 * i for i in row]
#             b = -model.getRhs(cons)
#         constraint_matrix.append(row)
#         vec_b.append(b)
#     # Add variable bounds as constraints
#     # count_i = 0
#     for i, var in enumerate(variables):
#         lb = var.getLbGlobal()
#         ub = var.getUbGlobal()
#         # print(var_names[i], "lb:", lb, "ub:", ub)
#         if lb > -1e+20:  # Check if lower bound is not -infinity
#             row = [0] * len(variables)
#             row[i] = 1
#             constraint_matrix.append(row)
#             vec_b.append(lb)
#
#
#         if ub < 1e+20:  # Check if upper bound is not infinity
#             row = [0] * len(variables)
#             row[i] = -1
#             constraint_matrix.append(row)
#             vec_b.append(-ub)
#
#     # print("count_i:", count_i)
#     return np.array(constraint_matrix), np.array(vec_b)


def get_currLP(model):
    """
    Get the current LP relaxation of the model
    """
    vars = model.getLPColsData()
    cons = model.getLPRowsData()
    A = []
    b = []
    for i in cons:
        print(i)

    return A, b
def general_disjunction(A, b, c, zl_init, delta):
    """
     Do a binary search over a range of values for zl and
     solving model_sub in each iteration of the search
     one can obtain the maximum value of the lower bound up to a desired level of accuracy.
     The intial value of zl is the dual bound of the original problem
     delta is a small positive number

    """
    # Create a new model
    zl_low = zl_init
    zl_high = zl_init * 10
    print("zl_high:", zl_high)
    print("zl_high - zl_low:", zl_high - zl_low)

    while np.abs(zl_high - zl_low) > 1e-6:  # Adjust the tolerance as needed
        # Adjust the tolerance as needed
        zl = (zl_high + zl_low) / 2
        num_vars, num_cons = A.shape
        m = num_vars
        n = num_cons
        print("m:", m, "n:", n)
        model_sub = Model("sub")
        # Define vector variables
        p = [model_sub.addVar(f"p_{i}", lb=0) for i in range(m)]
        s_L = model_sub.addVar(f"s_L", lb=0)
        q = [model_sub.addVar(f"q_{i}", lb=0) for i in range(m)]
        s_R = model_sub.addVar(f"s_R", lb=0)
        pi = [model_sub.addVar(f"pi_{j}", vtype="I") for j in range(n)]
        pi0 = model_sub.addVar("pi0", vtype="I")

        # Add constraints
        # pA − s_Lc − π = 0
        for j in range(n):
            model_sub.addCons(quicksum(p[i] * A[i][j] for i in range(m)) - s_L * c[j] - pi[j] == 0)

        # pb − s_Lz_l − π0 ≥ δ
        model_sub.addCons(
            quicksum(p[i] * b[i] for i in range(m)) - s_L * zl - pi0 >= delta)

        # qA − s_Rc + π = 0
        for j in range(n):
            model_sub.addCons(quicksum(q[i] * A[i][j] for i in range(m)) - s_R * c[j] + pi[j] == 0)

        # qb − s_Rz_l + π0 ≥ δ − 1
        model_sub.addCons(
            quicksum(q[i] * b[i] for i in range(m)) - s_R * zl + pi0 >= delta - 1)

        model_sub.optimize()

        if model_sub.getStatus() == "optimal":

            print("Feasible solution found!")
            zl_low = zl  # Move zl_low up
            # Extract and print the solution if needed
            p_solution = [model_sub.getVal(p[i]) for i in range(m)]
            s_L_solution = model_sub.getVal(s_L)
            q_solution = [model_sub.getVal(q[i]) for i in range(m)]
            s_R_solution = model_sub.getVal(s_R)
            pi_solution = [model_sub.getVal(pi[j]) for j in range(n)]
            pi0_solution = model_sub.getVal(pi0)
            if pi0_solution != 0.0:
                print("pi_solution:", pi_solution)
                print("pi0_solution:", pi0_solution)

            else:
                print("pi0_solution:", pi0_solution)
                print("pi_solution:", pi_solution)
                print("p_solution:", p_solution)
                print("s_L_solution:", s_L_solution)
                print("q_solution:", q_solution)
                print("s_R_solution:", s_R_solution)
        else:
            zl_high = zl  # Move zl_high down

# model_0 = (Model("test01"))
# model_0.setHeuristics(SCIP_PARAMSETTING.OFF)
# model_0.setIntParam("presolving/maxrounds", 0)
# model_0.setLongintParam("limits/nodes", 10000)
# model_0.setRealParam("limits/gap", 0.01)
#
# model_0.readProblem("D:/scipoptsuite-9.1.0/scipoptsuite-9.1.0/scip/check/instances/MIP/bell5.mps")

# Usage example
me = Model("example")
x = me.addVar("x", vtype= "I", lb=-10, ub=10)
y = me.addVar("y", vtype= "I", lb=0, ub=20)
z = me.addVar("z", lb=0, ub=30)

me.addCons(x + 2 * y + 3 * z <= 100)
me.addCons(x + 2 * y + 6 * z >= 20)
me.addCons(x + 2 * y + 8 * z <= 500)

me.setObjective(3*x + 2*y + z)
me.relax()
# vars = me.getVars()



# obj_expr = me.getObjective()
# c = np.array([obj_expr[var] for var in me.getVars()])
# print(c.shape)
# with open("A_output.txt", "w") as file_A:
#     np.savetxt(file_A, A, fmt="%.8f")
#
# with open("b_output.txt", "w") as file_b:
#     np.savetxt(file_b, b, fmt="%.8f")
#
# with open("c_output.txt", "w") as file_c:
#     np.savetxt(file_c, c, fmt="%.8f")
# me.relax()
# me.optimize()

# zl_init = me.getLPObjVal()

# general_disjunction(A, b, c, zl_init, delta = 0.05)


