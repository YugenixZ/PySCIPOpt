from pyscipopt import Model, Branchrule, SCIP_RESULT, quicksum, SCIP_PARAMSETTING, Eventhdlr, SCIP_EVENTTYPE
from sklearn import manifold
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import networkx as nx
import os
import math
from time import time
import re

def get_best_solution_value(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'\*BEST SOLN:\s+([\d.]+)', line)
            if match:
                return float(match.group(1))
    return None

def get_constraint_matrix(model):
    cols_lp = model.getLPColsData()
    Rows = model.getLPRowsData()
    NonZ_col = [i.getCols() for i in Rows]
    NonZ_Coeff = [i.getVals() for i in Rows]
    c = []
    for i in cols_lp:
        objcoeff = i.getObjCoeff()
        assert isinstance(objcoeff, object)
        c.append(objcoeff)

    assert len(NonZ_col) == len(NonZ_Coeff)

    # Extract the constraint matrix A and vector b
    curr_A = []
    curr_b = []
    for i in range(len(Rows)):
        row = [0] * len(cols_lp)
        for j in range(len(NonZ_col[i])):
            assert len(NonZ_col[i]) == len(NonZ_Coeff[i])
            # pres_varname = NonZ_col[i][j].getVar().name.split("_", 1)[-1]
            pos_currCol = NonZ_col[i][j].getLPPos()
            temp_coeff = NonZ_Coeff[i][j]
            row[pos_currCol] = temp_coeff
        lhs = Rows[i].getLhs()
        rhs = Rows[i].getRhs()
        if lhs == rhs:
            temp_b = lhs
            curr_A.append(row)
            curr_b.append(temp_b)
            curr_A.append([-coeff for coeff in row])
            curr_b.append(-temp_b)
        elif lhs == -1e+20:
            temp_b = -rhs
            row = [-1 * coeff for coeff in row]
            curr_A.append(row)
            curr_b.append(temp_b)
        else:
            temp_b = lhs
            curr_A.append(row)
            curr_b.append(temp_b)

    # Add the bound of each col of curr LP to the constraint matrix A and vector b
    for i in cols_lp:
        lb = i.getLb()
        ub = i.getUb()
        if lb > -1e+20:
            row = [0] * len(cols_lp)
            # varname = i.getVar().name.split("_", 1)[-1]
            # row[curr_vars.index(varname)] = 1
            row[i.getLPPos()] = 1
            curr_A.append(row)
            curr_b.append(lb)
        if ub < 1e+20:
            row = [0] * len(cols_lp)
            # row[curr_vars.index(i.getVar().name.split("_", 1)[-1])] = -1
            row[i.getLPPos()] = -1
            curr_A.append(row)
            curr_b.append(-ub)

    return np.array(curr_A), np.array(curr_b), np.array(c)

def check_model_two(name, A, b, c, pi_solution, pi0_solution, n, m, condition, best_zl):
    model_ck = Model(name)
    x = [model_ck.addVar(f"x_{i}", lb=None) for i in range(n)]

    for j in range(m):
        model_ck.addCons(quicksum(A[j][i] * x[i] for i in range(n)) >= b[j])

    if condition == "pi0":
        model_ck.addCons(quicksum(x[i] * pi_solution[i] for i in range(n)) <= pi0_solution)

    elif condition == "pi0+1":
        model_ck.addCons(quicksum(x[i] * pi_solution[i] for i in range(n)) >= pi0_solution + 1)

    model_ck.setObjective(quicksum(x[i] * c[i] for i in range(n)))
    model_ck.hideOutput()
    # set_numerics(model_ck)
    model_ck.optimize()

    if model_ck.getStatus() == "optimal":
        cx = model_ck.getObjVal()
        sols = model_ck.getBestSol()
        vars = model_ck.getVars()
        sol = []
        for i in range(n):
            sol.append(model_ck.getSolVal(sols, vars[i]))
        sol = np.array(sol)
        Ax = np.dot(A, sol)
        cons_1 = Ax - b
        for i in cons_1:
            assert i > -1e-6
        pix = np.dot(pi_solution, sol)
        if condition == "pi0":
            cons_2 = pix - pi0_solution
        else:
            cons_2 = pix - pi0_solution - 1

    return model_ck

def check_feasibility(model, model_org, Best_zl, n):

    if model.getStatus() == "optimal":
        obj_val = model.getObjVal()
        if model.getObjVal() - Best_zl > 1e-6:

            status = "updated_zl"
            # Get the fractional part of the integer/binary variables
            sol_ck = model.getBestSol()
            est = model.getObjVal()
        else:
            status = "unchanged_zl"
            est = 1e+20
            prob_name = model_org.getProbName()
            curr_node_num = model_org.getCurrentNode().getNumber()
            model.writeProblem(f"./Prob_obj_le_zl/{prob_name}_Node{curr_node_num}_with_zl{Best_zl}.lp")
            print("The objctive value is:", obj_val)
            return status, est
    else:
        status = "infeasible"
        est = 1e+20
        return status, est

    return status, est

def get_estimate(status, model):
    return model.getObjVal() if status == "updated_zl" else 1e+20

def general_disjunction(A, b, c, zl_init, M, k, delta, model):
    """
     Do a binary search over a range of values for zl and
     solving model_sub in each iteration of the search
     one can obtain the maximum value of the lower bound up to a desired level of accuracy.
     The intial value of zl is the dual bound of the original problem
     delta is a small positive number
    """

    try:
        # Initialize variables
        best_pi_solutions = []
        best_pi0_solutions = []
        estL_list = []
        estR_list = []
        Status_l = []
        Status_r = []
        zl_low = zl_init
        if zl_init > 0:
            zl_high = zl_init * 2
        elif zl_init < 0:
            zl_high = zl_init / 2
        else:
            zl_high = 1

        feasible_zl = []
        while np.abs(zl_high - zl_low) > 1e-6:
            zl = (zl_high + zl_low) * 0.5
            m, n = A.shape  # m is the number of rows and n is the number of columns
            model_sub = Model("sub")

            # Define vector variables
            p = [model_sub.addVar(f"p_{i}", lb=0) for i in range(m)]
            s_L = model_sub.addVar(f"s_L", lb=0)
            q = [model_sub.addVar(f"q_{i}", lb=0) for i in range(m)]
            s_R = model_sub.addVar(f"s_R", lb=0)
            pi_plus = [model_sub.addVar(f"pi_plus_{j}", vtype="I", lb=0, ub=M) for j in range(n)]
            pi_minus = [model_sub.addVar(f"pi_minus_{j}", vtype="I", lb=0, ub=M) for j in range(n)]
            pi0 = model_sub.addVar("pi0", vtype="I", lb=None)

            # pA − s_Lc − (π_plus - π_minus) = 0
            for j in range(n):
                model_sub.addCons(
                    quicksum(p[i] * A[i][j] for i in range(m)) - s_L * c[j] - pi_plus[j] + pi_minus[j] == 0)

            # pb − s_Lz_l − π0 ≥ δ
            model_sub.addCons(quicksum(p[i] * b[i] for i in range(m)) - s_L * zl - pi0 >= delta)

            # qA − s_Rc + (π_plus - π_minus) = 0
            for j in range(n):
                model_sub.addCons(
                    quicksum(q[i] * A[i][j] for i in range(m)) - s_R * c[j] + pi_plus[j] - pi_minus[j] == 0)

            # qb − s_Rz_l + π0 ≥ δ − 1
            model_sub.addCons(
                quicksum(q[i] * b[i] for i in range(m)) - s_R * zl + pi0 >= delta - 1)

            # Add the constraint ∑ abs(π+_i - π-_i) ≤ k
            model_sub.addCons(quicksum(pi_plus[i] + pi_minus[i] for i in range(n)) <= k)

            # add π0 < πx∗ < π0 + 1 if x∗ is known to be a fractional optimal solution of the LP relaxation of the original problem
            status_LP = model.getLPSolstat()
            if status_LP == 1:
                x_star = []  # Get the solution of the curr LP
                epsilon = 1e-4
                Cols = model.getLPColsData()
                for col in Cols:
                    v_lp = col.getVar()
                    x_star.append(model.getSolVal(None, v_lp))

                model_sub.addCons(pi0 <= quicksum((pi_plus[i] - pi_minus[i]) * x_star[i] for i in range(n)) - epsilon)
                model_sub.addCons(pi0 >= quicksum((pi_plus[i] - pi_minus[i]) * x_star[i] for i in range(n)) + epsilon - 1)
            # model_sub.hideOutput()
            model_sub.setRealParam("limits/time", 1000)
            model_sub.optimize()

            if model_sub.getStatus() == "optimal":
                pi_solution = np.array([model_sub.getVal(pi_plus[i]) - model_sub.getVal(pi_minus[i]) for i in range(n)])
                pi0_solution = model_sub.getVal(pi0)
                assert model_sub.isFeasIntegral(pi0_solution)
                for i in pi_solution:
                    assert model_sub.isFeasIntegral(i)

                ck_model_l = check_model_two("check_model_left", A, b, c, pi_solution, pi0_solution, n, m, "pi0", zl)
                ck_model_r = check_model_two("check_model_right", A, b, c, pi_solution, pi0_solution, n, m, "pi0+1", zl)

                status_l, est_l = check_feasibility(ck_model_l, model, zl, n)
                status_r, est_r = check_feasibility(ck_model_r, model, zl, n)

                if status_l == "updated_zl" or status_r == "updated_zl":
                    feasible_zl.append(zl)
                    best_pi_solutions.append(pi_solution)
                    best_pi0_solutions.append(pi0_solution)
                    Status_l.append(status_l)
                    Status_r.append(status_r)
                    zl_low = zl
                    if status_l == "updated_zl" and status_r != "updated_zl":
                        estL_list.append(est_l)
                        estR_list.append(1e+20)
                    elif status_r == "updated_zl" and status_l != "updated_zl":
                        estR_list.append(est_r)
                        estL_list.append(1e+20)
                    elif status_r == "updated_zl" and status_l == "updated_zl":
                        estL_list.append(est_l)
                        estR_list.append(est_r)

                elif status_l == "infeasible" and status_r == "infeasible":
                    feasible_zl.append(zl)
                    best_pi_solutions.append(pi_solution)
                    best_pi0_solutions.append(pi0_solution)
                    Status_l.append(status_l)
                    Status_r.append(status_r)
                    zl_high = zl
                else:
                    zl_high = zl
                #
                # elif status_r == "infeasible" and status_l != "updated_zl":
                #     feasible_zl.append(zl)
                #     best_pi_solutions.append(pi_solution)
                #     best_pi0_solutions.append(pi0_solution)
                #     Status_l.append(status_l)
                #     Status_r.append(status_r)
                #     zl_high = zl
                #
                # elif status_l == "unchanged_zl" and status_r == "unchanged_zl":
                #     ck_model_r.writeProblem(f"./Prob_obj_ge_zl/{model.getProbName()}_Node{model.getCurrentNode().getNumber()}_right.lp")
                #     ck_model_l.writeProblem(f"./Prob_obj_ge_zl/{model.getProbName()}_Node{model.getCurrentNode().getNumber()}_left.lp")
                #     print("Both models‘ lower bounds are less than the corresponding zl, Farkas' lemma is violated. The problems are written to the file.")
            else:
                zl_high = zl

        assert len(feasible_zl) == len(best_pi_solutions) == len(best_pi0_solutions)

        if len(feasible_zl) == 0:
            result = [None, None, None, None, None]
        else:
            best_zl = np.max(feasible_zl)
            idx_zl = feasible_zl.index(best_zl)
            best_pi_solution = best_pi_solutions[idx_zl]
            best_pi0_solution = best_pi0_solutions[idx_zl]
            data_l = [estL_list[idx_zl], Status_l[idx_zl]]
            data_r = [estR_list[idx_zl], Status_r[idx_zl]]
            result = [best_zl, best_pi_solution, best_pi0_solution, data_l, data_r]

    except Exception as e:
        print(f"An error occurred: {e}")
        result = [None, None, None, None, None]
        return result

    return result

def test_model_Abc(model, A, b, c, Cols_lp):
    # Get the LP solution for testing
    z_lp = model.getLPObjVal()
    solution = []
    for col in Cols_lp:
        v = col.getVar()
        solution.append(v.getLPSol())

    Ax = A @ solution
    # Assert tha Ax >= b
    for idx in range(len(b)):
        assert Ax[idx] - b[idx] > -1e-6, f"Constraint violation at index {idx}: Ax[i] = {Ax[idx]}, b[i] = {b[idx]}"
        # print("A[i]:", A[i])
    cx = np.dot(c, solution)
    assert np.abs(cx - z_lp) < 1e-6, f"Objective violation: cx = {cx}, zl = {z_lp}"

    model_test = Model("test")
    m = A.shape[0]
    n = A.shape[1]
    x = [model_test.addVar(f"x_{i}", lb=None) for i in range(n)]
    for j in range(m):
        model_test.addCons(quicksum(A[j][i] * x[i] for i in range(n)) >= b[j])
    model_test.setObjective(quicksum(x[i] * c[i] for i in range(n)))

    model_test.setIntParam("presolving/maxrestarts", 0)
    model_test.setIntParam("presolving/maxrounds", 0)
    model_test.setParam("estimation/restarts/restartpolicy", "n")

    model_test.setSeparating(SCIP_PARAMSETTING.OFF)
    model_test.setPresolve(SCIP_PARAMSETTING.OFF)
    model_test.setHeuristics(SCIP_PARAMSETTING.OFF)
    model_test.hideOutput()
    model_test.optimize()
    if model_test.getStatus() == "optimal":
        print("test model objective value:", model_test.getObjVal())
        print("original model objective value:", model.getLPObjVal())
        # assert model_test.getObjVal() == model.getObjVal()
    else:
        print("A, b, c are not correct")

    return 0


class LPstatEventhdlr(Eventhdlr):
    """PySCIPOpt Event handler to collect data on LP events."""

    transvars = {}

    def collectNodeInfo(self, firstlp=True):
        objval = self.model.getSolObjVal(None)
        if abs(objval) >= self.model.infinity():
            return

        LPsol = {}
        if self.transvars == {}:
            self.transvars = self.model.getVars(transformed=True)
        for var in self.transvars:
            solval = self.model.getSolVal(None, var)
            # store only solution values above 1e-6
            if abs(solval) > 1e-6:
                LPsol[var.name] = self.model.getSolVal(None, var)

        # skip duplicate nodes
        # if self.nodelist and LPsol == self.nodelist[-1].get("LPsol"):
        #     return
        node = self.model.getCurrentNode()
        if node.getNumber() != 1:
            parentnode = node.getParent()
            parent = parentnode.getNumber()
        else:
            parent = 1
        depth = node.getDepth()

        age = self.model.getNNodes()
        condition = math.log10(self.model.getCondition())
        iters = self.model.lpiGetIterations()
        pb = self.model.getPrimalbound()
        if pb >= self.model.infinity():
            pb = None

        nodedict = {
            "number": node.getNumber(),
            "LPsol": LPsol,
            "objval": objval,
            "parent": parent,
            "age": age,
            "depth": depth,
            "first": firstlp,
            "condition": condition,
            "iterations": iters,
            # "variables": self.model.getNVars(),
            # "constraints": self.model.getNConss(),
            "rows": self.model.getNLPRows(),
            "primalbound": pb,
            "dualbound": self.model.getDualbound(),
            "time": self.model.getSolvingTime()
        }
        # skip 0-iterations LPs (duplicates?)
        if firstlp:
            self.nodelist.append(nodedict)
        elif iters > 0:
            prevevent = self.nodelist[-1]
            if nodedict["number"] == prevevent["number"] and not prevevent["first"]:
                # overwrite data from previous LP event
                self.nodelist[-1] = nodedict
            else:
                self.nodelist.append(nodedict)

    def eventexec(self, event):
        if event.getType() == SCIP_EVENTTYPE.FIRSTLPSOLVED:
            self.collectNodeInfo(firstlp=True)
        elif event.getType() == SCIP_EVENTTYPE.LPSOLVED:
            self.collectNodeInfo(firstlp=False)
        else:
            print("unexpected event:" + str(event))
        return {}

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.LPEVENT, self)

class TreeD:
    """
    Draw a visual representation of the branch-and-cut tree of SCIP for
    a particular instance using spatial dissimilarities of the node LP solutions.

    Attributes:
        scip_settings (list of (str, value)): list of optional SCIP settings to use when solving the instance
        scip_model (scip.Model): optional SCIP model, can be used to use instances generated with PySCIPOpt or models
        with user-defined plugins.
        transformation (sr): type of transformation to generate 2D data points ('tsne, 'mds')
        showcuts (bool): whether to show nodes/solutions that originate from cutting rounds
        color (str): data to use for colorization of nodes ('age', 'depth', 'condition')
        colorscale (str): type of colorization, e.g. 'Viridis', 'Portland'
        colorbar (bool): whether to show the colorbar
        title (bool): show/hide title of the plot
        showlegend (bool): show/hide logend of the plot
        fontsize (str): fixed fontsize or 'auto'
        nodesize (int): size of tree nodes
        weights (str): type of weights for pysal, e.g. 'knn'
        kernelfunction (str): type of kernelfunction for distance metrics, e.g. 'triangular'
        knn_k (int): number of k-nearest neighbors
        fig (object): handler for generated figure
        df (Dataframe): storage of tree information
        div (str): html for saving a plot.ly object to be included as div

    Dependencies:
     - PySCIPOpt to solve the instance and generate the necessary tree data
     - Plot.ly to draw the 3D visualization
     - pandas to organize the collected data
    """

    def __init__(self, **kwargs):
        self.probpath = kwargs.get("probpath", "")
        self.scip_model = kwargs.get("scip_model", None)
        self.scip_settings = [("limits/totalnodes", kwargs.get("nodelimit", 500))]
        self.setfile = kwargs.get("setfile", None)
        self.transformation = kwargs.get("transformation", "mds")
        self.showcuts = kwargs.get("showcuts", True)
        self.verbose = kwargs.get("verbose", True)
        self.color = "age"
        self.colorscale = "Portland"
        self.colorbar = kwargs.get("colorbar", True)
        self.title = kwargs.get("title", True)
        self.showlegend = kwargs.get("showlegend", True)
        self.showbuttons = kwargs.get("showbuttons", True)
        self.showslider = kwargs.get("showslider", True)
        self.fontsize = None
        self.nodesize = 5
        self.weights = "knn"
        self.kernelfunction = "triangular"
        self.knn_k = 2
        self.fig = None
        self.df = None
        self.div = None
        self.include_plotlyjs = "cdn"
        self.nxgraph = nx.Graph()
        self.stress = None
        self.start_frame = kwargs.get("start_frame", 1)

    def transform(self):
        """compute transformations of LP solutions into 2-dimensional space"""
        # df = pd.DataFrame(self.nodelist, columns = ['LPsol'])
        df = self.df["LPsol"].apply(pd.Series).fillna(value=0)
        if self.transformation == "tsne":
            mf = manifold.TSNE(n_components=2)
        elif self.transformation == "lle":
            mf = manifold.LocallyLinearEmbedding(n_components=2)
        elif self.transformation == "ltsa":
            mf = manifold.LocallyLinearEmbedding(n_components=2, method="ltsa")
        elif self.transformation == "spectral":
            mf = manifold.SpectralEmbedding(n_components=2)
        else:
            mf = manifold.MDS(n_components=2)

        if self.verbose:
            print("transforming LP solutions", end="...")
            start = time()
        start = time()
        xy = mf.fit_transform(df)
        if self.verbose:
            print(f"✔, time: {time() - start:.2f} seconds")

        try:
            self.stress = mf.stress_  # not available with all transformations
        except:
            print("no stress information available for {self.transformation} transformation")

        self.df["x"] = xy[:, 0]
        self.df["y"] = xy[:, 1]

    # def performSpatialAnalysis(self):
    #     """compute spatial correlation between LP solutions and their condition numbers"""
    #     import pysal

    #     df = pd.DataFrame(self.nodelist, columns=["LPsol", "condition"])
    #     lpsols = df["LPsol"].apply(pd.Series).fillna(value=0)
    #     if self.weights == "kernel":
    #         weights = pysal.Kernel(lpsols, function=self.kernelfunction)
    #     else:
    #         weights = pysal.knnW_from_array(lpsols, k=self.knn_k)
    #     self.moran = pysal.Moran(df["condition"].tolist(), weights)

    def _generateEdges(self, separate_frames=False):
        """Generate edge information corresponding to parent information in df

        :param separate_frames: whether to generate separate edge sets for each node age
                                (used for tree growth animation)
        """

        # 3D edges
        Xe = []
        Ye = []
        Ze = []

        if not "x" in self.df or not "y" in self.df:
            self.df["x"] = 0
            self.df["y"] = 0

        symbol = []

        if self.showcuts:
            self.nxgraph.add_nodes_from(self.df["id"])
            for index, curr in self.df.iterrows():
                if curr["first"]:
                    symbol += ["circle"]
                    # skip root node
                    if curr["number"] == 1:
                        continue
                    # found first LP solution of a new child node
                    # parent is last LP of parent node
                    parent = self.df[self.df["number"] == curr["parent"]].iloc[-1]
                else:
                    # found an improving LP solution at the same node as before
                    symbol += ["diamond"]
                    parent = self.df.iloc[index - 1]

                Xe += [parent["x"], curr["x"], None]
                Ye += [parent["y"], curr["y"], None]
                Ze += [parent["objval"], curr["objval"], None]
                self.nxgraph.add_edge(parent["id"], curr["id"])
        else:
            self.nxgraph.add_nodes_from(self.df["id"])
            for index, curr in self.df.iterrows():
                symbol += ["circle"]
                if curr["number"] == 1:
                    continue
                parent = self.df[self.df["number"] == curr["parent"]].iloc[-1]
                Xe += [parent["x"], curr["x"], None]
                Ye += [parent["y"], curr["y"], None]
                Ze += [parent["objval"], curr["objval"], None]
                self.nxgraph.add_edge(parent["id"], curr["id"])

        self.df["symbol"] = symbol
        self.Xe = Xe
        self.Ye = Ye
        self.Ze = Ze

    def _create_nodes_and_projections(self):
        colorbar = go.scatter3d.marker.ColorBar(title="", thickness=10, x=-0.05)
        marker = go.scatter3d.Marker(
            symbol=self.df["symbol"],
            size=self.nodesize,
            color=self.df["age"],
            colorscale=self.colorscale,
            colorbar=colorbar if self.colorbar else None,
        )
        node_object = go.Scatter3d(
            x=self.df["x"],
            y=self.df["y"],
            z=self.df["objval"],
            mode="markers+text",
            marker=marker,
            hovertext=self.df["number"],
            hovertemplate="LP obj: %{z}<br>node number: %{hovertext}<br>%{marker.color}",
            hoverinfo="z+text+name",
            opacity=0.7,
            name="LP solutions",
        )
        proj_object = go.Scatter3d(
            x=self.df["x"],
            y=self.df["y"],
            z=self.df["objval"],
            mode="markers+text",
            marker=marker,
            hovertext=self.df["number"],
            hoverinfo="z+text+name",
            opacity=0.0,
            projection=dict(z=dict(show=True)),
            name="projection of LP solutions",
            visible="legendonly",
        )
        return node_object, proj_object

    def _create_nodes_frames(self):
        colorbar = go.scatter3d.marker.ColorBar(title="", thickness=10, x=-0.05)
        marker = go.scatter3d.Marker(
            symbol=self.df["symbol"],
            size=self.nodesize,
            color=self.df["age"],
            colorscale=self.colorscale,
            colorbar=colorbar,
        )

        frames = []
        sliders_dict = dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue={"prefix": "Age:", "visible": True, "xanchor": "right", },
            len=0.9,
            x=0.05,
            y=0.1,
            steps=[],
        )

        # get start and end points for bound line plots
        min_x = min(self.df["x"])
        max_x = max(self.df["x"])
        min_y = min(self.df["y"])
        max_y = max(self.df["y"])

        maxage = max(self.df["age"])
        for a in np.linspace(1, maxage, min(200, maxage)):
            a = int(a)
            adf = self.df[self.df["age"] <= a]
            node_object = go.Scatter3d(
                x=adf["x"],
                y=adf["y"],
                z=adf["objval"],
                mode="markers+text",
                marker=marker,
                hovertext=adf["number"],
                # hovertemplate="LP obj: %{z}<br>node number: %{hovertext}<br>%{marker.color}",
                hoverinfo="z+text+name",
                opacity=0.7,
                name="LP Solutions",
            )

            primalbound = go.Scatter3d(
                x=[min_x, min_x, max_x, max_x, min_x],
                y=[min_y, max_y, max_y, min_y, min_y],
                z=[adf["primalbound"].iloc[-1]] * 5,
                mode="lines",
                line=go.scatter3d.Line(width=5),
                hoverinfo="name+z",
                name="Primal Bound",
                opacity=0.5,
            )
            dualbound = go.Scatter3d(
                x=[min_x, min_x, max_x, max_x, min_x],
                y=[min_y, max_y, max_y, min_y, min_y],
                z=[adf["dualbound"].iloc[-1]] * 5,
                mode="lines",
                line=go.scatter3d.Line(width=5),
                hoverinfo="name+z",
                name="Dual Bound",
                opacity=0.5,
            )

            frames.append(
                go.Frame(data=[node_object, primalbound, dualbound], name=str(a))
            )

            slider_step = {
                "args": [
                    [a],
                    {
                        "frame": {"redraw": True, "restyle": False},
                        "fromcurrent": True,
                        "mode": "immediate",
                    },
                ],
                "label": a,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        return frames, sliders_dict

    def _create_nodes_frames_2d(self):
        colorbar = go.scatter.marker.ColorBar(title="", thickness=10, x=-0.05)
        marker = go.scatter.Marker(
            symbol=self.df["symbol"],
            size=self.nodesize * 2,
            color=self.df["age"],
            colorscale=self.colorscale,
            colorbar=colorbar,
        )

        frames = []
        sliders_dict = dict(
            active=self.start_frame - 1,
            yanchor="top",
            xanchor="left",
            currentvalue={"prefix": "Age:", "visible": True, "xanchor": "right", },
            len=0.9,
            x=0.05,
            y=0.1,
            steps=[],
        )

        # get start and end points for bound line plots
        xmin = min([self.pos2d[k][0] for k in self.pos2d])
        xmax = max([self.pos2d[k][0] for k in self.pos2d])

        maxage = max(self.df["age"])
        for a in np.linspace(1, maxage, min(200, maxage)):
            a = int(a)
            adf = self.df[self.df["age"] <= a]
            node_object = go.Scatter(
                x=[self.pos2d[k][0] for k in adf["id"]],
                # y=[self.pos2d[k][1] for k in adf["id"]],
                y=[self.df["objval"][k] for k in adf["id"]],
                mode="markers",
                marker=marker,
                # hovertext=[
                #     f"LP obj: {adf['objval'].iloc[i]:.3f}\
                #     <br>node number: {adf['number'].iloc[i]}\
                #     <br>node age: {adf['age'].iloc[i]}\
                #     <br>depth: {adf['depth'].iloc[i]}\
                #     <br>LP cond: {adf['condition'].iloc[i]:.1f}\
                #     <br>iterations: {adf['iterations'].iloc[i]}"
                #     for i in range(len(adf))
                # ],
                hoverinfo="text+name",
                opacity=0.7,
                name="LP Solutions",
            )
            primalbound = go.Scatter(
                x=[xmin, xmax],
                y=2 * [adf["primalbound"].iloc[-1]],
                mode="lines",
                opacity=0.5,
                name="Primal Bound",
            )
            dualbound = go.Scatter(
                x=[xmin, xmax],
                y=2 * [adf["dualbound"].iloc[-1]],
                mode="lines",
                opacity=0.5,
                name="Dual Bound",
            )

            frames.append(
                go.Frame(data=[node_object, primalbound, dualbound], name=str(a))
            )

            slider_step = {
                "args": [
                    [a],
                    {
                        "frame": {"redraw": True, "restyle": False},
                        "fromcurrent": True,
                        "mode": "immediate",
                    },
                ],
                "label": a,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        return frames, sliders_dict

    def updatemenus(self):
        return list(
            [
                dict(
                    buttons=list(
                        [
                            dict(
                                label="Node Age",
                                method="restyle",
                                args=[
                                    {
                                        "marker.color": [self.df["age"]],
                                        "marker.cauto": min(self.df["age"]),
                                        "marker.cmax": max(self.df["age"]),
                                    }
                                ],
                            ),
                            dict(
                                label="Tree Depth",
                                method="restyle",
                                args=[
                                    {
                                        "marker.color": [self.df["depth"]],
                                        "marker.cauto": min(self.df["depth"]),
                                        "marker.cmax": max(self.df["depth"]),
                                    }
                                ],
                            ),
                            dict(
                                label="LP Condition (log 10)",
                                method="restyle",
                                args=[
                                    {
                                        "marker.color": [self.df["condition"]],
                                        "marker.cmin": 1,
                                        "marker.cmax": 20,
                                    }
                                ],
                            ),
                            dict(
                                label="LP Iterations",
                                method="restyle",
                                args=[
                                    {
                                        "marker.color": [self.df["iterations"]],
                                        "marker.cauto": min(self.df["iterations"]),
                                        "marker.cmax": max(self.df["iterations"]),
                                    }
                                ],
                            ),
                            # dict(
                            #     label="logarithmic obj",
                            #     method="relayout",
                            #     args=[
                            #         {
                            #             "yaxis.type": 'log'
                            #         }
                            #     ],
                            # ),
                            # dict(
                            #     label="linear obj",
                            #     method="relayout",
                            #     args=[
                            #         {
                            #             "yaxis.type": 'linear'
                            #         }
                            #     ],
                            # ),
                        ]
                    ),
                    direction="down",
                    showactive=True,
                    type="buttons",
                ),
                dict(
                    buttons=list(
                        [
                            dict(
                                label="▶",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 50, "redraw": True, },
                                        "fromcurrent": True,
                                    },
                                ],
                            ),
                            dict(
                                label="◼",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ]
                    ),
                    direction="left",
                    yanchor="top",
                    xanchor="right",
                    showactive=True,
                    type="buttons",
                    x=0,
                    y=0,
                ),
            ]
        )

    def draw(self, path):
        """Draw the tree, depending on the mode"""

        self.transform()

        if self.verbose:
            print("generating 3D objects", end="...")
            start = time()
        self._generateEdges()
        nodes, nodeprojs = self._create_nodes_and_projections()
        frames, sliders = self._create_nodes_frames()

        edges = go.Scatter3d(
            x=self.Xe,
            y=self.Ye,
            z=self.Ze,
            mode="lines",
            line=go.scatter3d.Line(color="rgb(75,75,75)", width=2),
            hoverinfo="none",
            name="Edges",
        )

        min_x = min(self.df["x"])
        max_x = max(self.df["x"])
        min_y = min(self.df["y"])
        max_y = max(self.df["y"])

        primalbound = go.Scatter3d(
            x=[min_x, min_x, max_x, max_x, min_x],
            y=[min_y, max_y, max_y, min_y, min_y],
            z=5 * [self.df["primalbound"].iloc[-1]],
            mode="lines",
            line=go.scatter3d.Line(width=5),
            hoverinfo="name+z",
            opacity=0.5,
            name="Primal Bound",
        )
        dualbound = go.Scatter3d(
            x=[min_x, min_x, max_x, max_x, min_x],
            y=[min_y, max_y, max_y, min_y, min_y],
            z=5 * [self.df["dualbound"].iloc[-1]],
            mode="lines",
            line=go.scatter3d.Line(width=5),
            hoverinfo="name+z",
            opacity=0.5,
            name="Dual Bound",
        )
        optval = go.Scatter3d(
            x=[min_x, min_x, max_x, max_x, min_x],
            y=[min_y, max_y, max_y, min_y, min_y],
            z=[self.optval] * 5,
            mode="lines",
            line=go.scatter3d.Line(width=5),
            hoverinfo="name+z",
            name="Optimum",
            opacity=0.5,
        )

        xaxis = go.layout.scene.XAxis(
            showticklabels=False,
            title="X",
            backgroundcolor="white",
            gridcolor="lightgray",
        )
        yaxis = go.layout.scene.YAxis(
            showticklabels=False,
            title="Y",
            backgroundcolor="white",
            gridcolor="lightgray",
        )
        zaxis = go.layout.scene.ZAxis(
            title="Objective value", backgroundcolor="white", gridcolor="lightgray"
        )
        scene = go.layout.Scene(xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)

        if self.title:
            title = f"TreeD: {self.probname} ({self.scipversion}, {self.status})"
        else:
            title = ""

        filename = path + "TreeD_" + self.probname + ".html"

        camera = dict(eye=dict(x=1.5, y=1.3, z=0.5))
        layout = go.Layout(
            title=title,
            font=dict(size=self.fontsize),
            font_family="Fira Sans",
            autosize=True,
            # width=900,
            # height=600,
            showlegend=self.showlegend,
            hovermode="closest",
            scene=scene,
            scene_camera=camera,
            template="none",
        )

        if self.showbuttons:
            layout["updatemenus"] = self.updatemenus()
        if self.showslider:
            layout["sliders"] = [sliders]

        self.fig = go.Figure(
            data=[nodes, primalbound, dualbound, optval, nodeprojs, edges],
            layout=layout,
            frames=frames,
        )

        self.fig.write_html(file=filename, include_plotlyjs=self.include_plotlyjs)

        # generate html code to include into a website as <div>
        # self.div = self.fig.write_html(
        #     file=filename, include_plotlyjs=self.include_plotlyjs, full_html=False
        # )

        if self.verbose:
            print(f"✔, time: {time() - start:.2f} seconds")

        return self.fig

    def draw2d(self, path):
        """Draw the 2D tree"""
        self._generateEdges()
        self.hierarchy_pos()
        frames, sliders = self._create_nodes_frames_2d()

        start_frame = self.df[self.df["age"] <= self.start_frame]

        Xv = [self.pos2d[k][0] for k in self.df["id"]]
        # Yv = [self.pos2d[k][1] for k in self.df["id"]]
        # Yv = self.df["objval"]
        Xed = []
        Yed = []
        for edge in self.nxgraph.edges:
            Xed += [self.pos2d[edge[0]][0], self.pos2d[edge[1]][0], None]
            # Yed += [self.pos2d[edge[0]][1], self.pos2d[edge[1]][1], None]
            Yed += [self.df["objval"][edge[0]], self.df["objval"][edge[1]], None]

        colorbar = go.scatter.marker.ColorBar(title="", thickness=10, x=-0.05)
        marker = go.scatter.Marker(
            symbol=self.df["symbol"],
            size=self.nodesize * 2,
            color=self.df["age"],
            colorscale=self.colorscale,
            colorbar=colorbar if self.colorbar else None,
        )

        edges = go.Scatter(
            x=Xed,
            y=Yed,
            mode="lines",
            line=dict(color="rgb(75,75,75)", width=1),
            hoverinfo="none",
            name="Edges",
        )
        nodes = go.Scatter(
            x=[self.pos2d[k][0] for k in start_frame["id"]],
            y=start_frame["objval"],
            name="LP solutions",
            mode="markers",
            marker=marker,
            hovertext=[
                f"LP obj: {self.df['objval'].iloc[i]:.3f}\
                <br>node number: {self.df['number'].iloc[i]}\
                <br>node age: {self.df['age'].iloc[i]}\
                <br>depth: {self.df['depth'].iloc[i]}\
                <br>LP cond: {self.df['condition'].iloc[i]:.1f}\
                <br>iterations: {self.df['iterations'].iloc[i]}"
                for i in range(len(self.df))
            ],
            hoverinfo="text+name",
        )

        xmin = min(Xv)
        xmax = max(Xv)
        primalbound = go.Scatter(
            x=[xmin, xmax],
            y=2 * [start_frame["primalbound"].iloc[-1]],
            mode="lines",
            opacity=0.5,
            name="Primal Bound",
        )
        dualbound = go.Scatter(
            x=[xmin, xmax],
            y=2 * [start_frame["dualbound"].iloc[-1]],
            mode="lines",
            opacity=0.5,
            name="Dual Bound",
        )
        optval = go.Scatter(
            x=[xmin, xmax],
            y=2 * [self.optval],
            mode="lines",
            opacity=0.5,
            name="Optimum",
        )

        margin = 0.05 * xmax
        xaxis = go.layout.XAxis(
            title="",
            visible=False,
            range=[xmin - margin, xmax + margin],
            autorange=False,
        )
        yaxis = go.layout.YAxis(
            title="Objective value", visible=True, side="right", position=0.98
        )

        if self.title:
            title = f"Tree 2D: {self.probname} ({self.scipversion}, {self.status})"
        else:
            title = ""
        filename = path + "Tree_2D_" + self.probname + ".html"

        layout = go.Layout(
            title=title,
            font=dict(size=self.fontsize),
            font_family="Fira Sans",
            autosize=True,
            template="none",
            showlegend=self.showlegend,
            hovermode="closest",
            xaxis=xaxis,
            yaxis=yaxis,
        )

        if self.showbuttons:
            layout["updatemenus"] = self.updatemenus()
        layout["sliders"] = [sliders]

        self.fig2d = go.Figure(
            data=[nodes, primalbound, dualbound, optval, edges],
            layout=layout,
            frames=frames,
        )

        self.fig2d.write_html(file=filename, include_plotlyjs=self.include_plotlyjs)

        return self.fig2d

    def solve(self, branchingrule, t, bestsol):
        """Solve the instance and collect and generate the tree data"""

        self.nodelist = []

        if self.scip_model:
            self.probname = self.scip_model.getProbName()
            model = self.scip_model
        else:
            self.probname = os.path.splitext(os.path.basename(self.probpath))[0]
            model = Model(f"{self.probname}_model")

        if self.verbose:
            model.redirectOutput()
        else:
            model.hideOutput()

        eventhdlr = LPstatEventhdlr()
        eventhdlr.nodelist = self.nodelist
        model.includeEventhdlr(
            eventhdlr, "LPstat", "generate LP statistics after every LP event"
        )
        model.readProblem(self.probpath)
        if self.setfile:
            model.readParams(self.setfile)

        # Adjust presolving settings
        model.setIntParam("presolving/maxrestarts", 0)
        model.setIntParam("presolving/maxrounds", 0)
        model.setParam("estimation/restarts/restartpolicy", "n")
        model.readSol(bestsol)
        # set_numerics(model)

        # Adjust LP settings
        model.setSeparating(SCIP_PARAMSETTING.OFF)
        model.setPresolve(SCIP_PARAMSETTING.OFF)
        model.setHeuristics(SCIP_PARAMSETTING.OFF)

        # Time limit for solving the problem
        model.setRealParam("limits/time", 8*t)

        if branchingrule == "generaldisjunction":
            mybranching = MyBranching(model)
            model.includeBranchrule(mybranching, "test branch", "test branching and probing and lp functions",
                                    priority=1000000, maxdepth=-1, maxbounddist=1)
        elif branchingrule == "fullstrong":
            model.setIntParam("branching/fullstrong/priority", 1000000)
        elif branchingrule == "relpscost":
            model.setIntParam("branching/relpscost/priority", 1000000)
        elif branchingrule == "pscost":
            model.setIntParam("branching/pscost/priority", 1000000)
        elif branchingrule == "mostinf":
            model.setIntParam("branching/mostinf/priority", 1000000)

        for setting in self.scip_settings:
            model.setParam(setting[0], setting[1])

        if self.verbose:
            print("optimizing problem", end="... ")
            start = time()
        try:
            model.optimize()
        except:
            print("optimization failed")

        if self.verbose:
            print(f"{model.getStatus()}, time: {time() - start:.2f} seconds")

        self.scipversion = "SCIP " + str(model.version())
        # self.scipversion = self.scipversion[:-1]+'.'+self.scipversion[-1]

        self.status = model.getStatus()
        if self.status == "optimal":
            self.optval = model.getObjVal()
        else:
            self.optval = None

        # print("performing Spatial Analysis on similarity of LP condition numbers")
        # self.performSpatialAnalysis()

        columns = self.nodelist[0].keys()
        self.df = pd.DataFrame(self.nodelist, columns=columns)

        # drop last data point of every node, since it's a duplicate
        # for n in self.df["number"]:
        #     seq = self.df[(self.df["first"] == False) & (self.df["number"] == n)]
        #     if len(seq) > 0:
        #         self.df.drop(
        #             index=seq.index[-1], inplace=True,
        #         )

        # merge solutions from cutting rounds into one node, preserving the latest
        # if self.showcuts:
        #     self.df = (
        #         self.df[self.df["first"] == False]
        #         .drop_duplicates(subset="age", keep="last")
        #         .reset_index()
        #     )
        if not self.showcuts:
            self.df = self.df[self.df["first"]].reset_index()

        self.df["id"] = self.df.index

    def hierarchy_pos(self, root=0, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """compute abstract node positions of the tree
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
        Licensed under Creative Commons Attribution-Share Alike
        """
        G = self.nxgraph
        if not nx.is_tree(G):
            # raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")
            self.pos2d = nx.kamada_kawai_layout(G)
            return

        def _hierarchy_pos(
                G,
                root,
                width=1.0,
                vert_gap=0.2,
                vert_loc=0,
                xcenter=0.5,
                pos=None,
                parent=None,
        ):
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(
                        G,
                        child,
                        width=dx,
                        vert_gap=vert_gap,
                        vert_loc=vert_loc - vert_gap,
                        xcenter=nextx,
                        pos=pos,
                        parent=root,
                    )
            return pos

        self.pos2d = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    def compute_distances(self):
        """compute all pairwise distances between the original LP solutions and the transformed points"""
        if self.df is None:
            return

        origdist = []
        transdist = []
        for i in range(len(self.df)):
            for j in range(i + 1, len(self.df)):
                origdist.append(
                    self.distance(self.df["LPsol"].iloc[i], self.df["LPsol"].iloc[j])
                )
                transdist.append(
                    self.distance(
                        self.df[["x", "y"]].iloc[i], self.df[["x", "y"]].iloc[j]
                    )
                )
        self.distances = pd.DataFrame()
        self.distances["original"] = origdist
        self.distances["transformed"] = transdist

    @staticmethod
    def distance(p1, p2):
        """euclidean distance between two coordinates (dict-like storage)"""
        dist = 0
        for k in set([*p1.keys(), *p2.keys()]):
            dist += (p1.get(k, 0) - p2.get(k, 0)) ** 2
        return math.sqrt(dist)

class MyBranching(Branchrule):

    def __init__(self, model):
        self.model = model

    def branchexeclp(self, allowaddcons):

        # Get the current node information
        curr_Node = self.get_information()
        zl_init = self.model.getLPObjVal()
        Cols_lp = self.model.getLPColsData()
        variables_lp = [c.getVar() for c in Cols_lp]

        # Extract the constraint matrix A and vector b
        A, b, c = get_constraint_matrix(self.model)

        # Check the constraint matrix and vector
        assert len(c) == A.shape[1]
        assert len(b) == A.shape[0]

        test_model_Abc(self.model, A, b, c, Cols_lp)

        delta = 0.05 #(np.sum(b)+ zl_init)* 1e-08
        M = 1
        k = 2
        zl_curr, pi_curr, pi0_curr, data_l, data_r = general_disjunction(A, b, c, zl_init, M, k, delta, self.model)

        # Create down children
        downprio = 1.0
        print(f"Rows of A on Node {curr_Node.getNumber()}:", A.shape[0])
        print(f"Columns of A on Node {curr_Node.getNumber()}:", A.shape[1])

        print("Allowaddcons:", allowaddcons)
        if data_l is None or data_r is None:
            print("Both children are not added, data is None")
            return {"result": SCIP_RESULT.DIDNOTFIND}

        elif data_l[1] == "updated_zl" and data_r[1] == "updated_zl":

            left_child = self.model.createChild(downprio, data_l[0])
            # add left constraint: pi * x <= pi0
            cons_l = self.model.createConsFromExpr(
                quicksum(pi_curr[i] * variables_lp[i] for i in range(len(variables_lp))) <= pi0_curr,
                'left' + str(curr_Node.getNumber()))
            # print("Left constraint pi:", pi_curr)
            # print("Left constraint pi0:", pi0_curr)
            self.model.addConsNode(left_child, cons_l)

            # create down child for cm2_status
            right_child = self.model.createChild(downprio, data_r[0])
            # add right constraint: pi * x >= pi0 + 1
            cons_r = self.model.createConsFromExpr(
                quicksum(pi_curr[i] * variables_lp[i] for i in range(len(variables_lp))) >= pi0_curr + 1,
                'right' + str(curr_Node.getNumber()))
            # print("Right constraint pi:", pi_curr)
            # print("Right constraint pi0:", pi0_curr)
            self.model.addConsNode(right_child, cons_r)

            print("Both children are added")
            return {"result": SCIP_RESULT.BRANCHED}

        elif data_l[1] == "infeasible" and data_r[1] != "updated_zl":

            print("Infeasible left child")
            return {"result": SCIP_RESULT.CUTOFF}
        elif data_l[1] != "updated_zl" and data_r[1] == "infeasible":

            print("Infeasible right child")
            return {"result": SCIP_RESULT.CUTOFF}

        elif data_l[1] == "updated_zl" and data_r[1] != "updated_zl" :

            child_node = self.model.createChild(downprio, data_l[0])
            # add left constraint: pi * x <= pi0
            cons_l = self.model.createConsFromExpr(
                quicksum(pi_curr[i] * variables_lp[i] for i in range(len(variables_lp))) <= pi0_curr,
                'left' + str(curr_Node.getNumber()))

            self.model.addConsNode(child_node, cons_l)
            # self.model.addConsLocal(cons_l)
            print("Only Left constraint added:")

            return {"result": SCIP_RESULT.BRANCHED}
            # return {"result": SCIP_RESULT.CONSADDED}

        elif data_r[1] == "updated_zl" and data_l[1] != "updated_zl":

            child_node = self.model.createChild(downprio, data_r[0])
            # add right constraint: pi * x >= pi0 + 1
            cons_r = self.model.createConsFromExpr(
                quicksum(pi_curr[i] * variables_lp[i] for i in range(len(variables_lp))) >= pi0_curr + 1,
                'right' + str(curr_Node.getNumber()))
            self.model.addConsNode(child_node, cons_r)
            print("Only Right constraint added:")
            # self.model.addConsLocal(cons_r)

            return {"result": SCIP_RESULT.BRANCHED}
            # return {"result": SCIP_RESULT.CONSADDED}
        else:
            print("Both children are not added")
            return {"result": SCIP_RESULT.DIDNOTFIND}

    def get_information(self):
        print("_____________________________________")
        print("Now starting branching")
        # Check if the added constraint is added to the node or not
        curr_Node = self.model.getCurrentNode()
        print("Current branching Node number:", curr_Node.getNumber())
        if curr_Node.getDepth() > 0:
            num_addedCons = curr_Node.getNAddedConss()
            addedCons = curr_Node.getAddedConss()

            if addedCons:
                print("Added constraint:", self.model.getRowLinear(addedCons[0]).getVals())
                print("Rhs:", self.model.getRhs(addedCons[0]))
                print("Lhs:", self.model.getLhs(addedCons[0]))
                print("Number of added constraints:", num_addedCons)

        return curr_Node


if __name__ == "__main__":

    branchingrule_list = ["generaldisjunction"]
    time_list = [1000, 100, 50]
    files = os.listdir("D:/scipoptsuite-8.1.0/res_log/sms/PySCIPOpt/tests/test_MIP/")
    mps_files = [f for f in files if f.endswith(".mps")]
    for mps_file in mps_files:
        print(mps_file)
        mps_path = os.path.join("D:/scipoptsuite-8.1.0/res_log/sms/PySCIPOpt/tests/test_MIP/", mps_file)
        # best_solution_value = get_best_solution_value(mps_path)
        best_sol_file = mps_path.replace(".mps", ".sol")
        for i in branchingrule_list:
            treed = TreeD(probpath=mps_path, showcuts=False, nodelimit=10000)
            treed.solve(i, 1000, best_sol_file)
            fig = treed.draw2d(path=f"./{i}_nodes_plots/")
            # fig.show()



    # files = os.listdir("D:/scipoptsuite-8.1.0/res_log/sms/PySCIPOpt/tests/test_MIP/tested")
    # mps_files = [f for f in files if f.endswith(".mps")]
    # for mps_file in mps_files:
    #     # Use the problem from files
    #     mf = Model("file")
    #     mf.setIntParam("presolving/maxrounds", 0)
    #
    #     mf.setHeuristics(SCIP_PARAMSETTING.OFF)
    #     mf.setSeparating(SCIP_PARAMSETTING.OFF)
    #     mf.setPresolve(SCIP_PARAMSETTING.OFF)
    #     print(f"Now testing {mps_file}")
    #     mps_path = "D:/scipoptsuite-8.1.0/res_log/sms/PySCIPOpt/tests/test_MIP/tested" + "/" + mps_file
    #     mf.readProblem(mps_path)
    #     mf.readSol(mps_path.replace(".mps", ".sol"))
    #     my_branchrule_1 = MyBranching(mf)
    #     mf.includeBranchrule(my_branchrule_1, "test branch", "test branching and probing and lp functions",
    #                          priority=10000000, maxdepth=-1, maxbounddist=1)
    #
    #     branchingrule_list = ["generaldisjunction", "fullstrong", "relpscost", "pscost", "mostinf"]
    #     mf.optimize()
    # # Create the model
    # model_kp = Model("Knapsack Problem")
    #
    # # Define the number of items
    # n_items = 20
    #
    # # Define the weights and values for each item
    # weights = [2, 3, 4, 5, 1, 6, 7, 8, 3, 4, 5, 6, 3, 4, 7, 6, 8, 2, 4, 3]
    # values = [12, 10, 25, 40, 15, 50, 30, 60, 10, 20, 30, 60, 10, 30, 40, 30, 50, 15, 25, 45]
    #
    # # Define the knapsack capacity
    # capacity = 50
    #
    # # Add decision variables (binary variables indicating whether an item is taken or not)
    # x = [model_kp.addVar(f"x_{i}", vtype="B") for i in range(n_items)]
    #
    # # Add the constraint to ensure the total weight does not exceed the capacity
    # model_kp.addCons(-sum(weights[i] * x[i] for i in range(n_items)) >= -capacity)
    #
    # # Set the objective function to maximize the total value
    # model_kp.setObjective(-sum(values[i] * x[i] for i in range(n_items)))
    #
    # model_kp.setHeuristics(SCIP_PARAMSETTING.OFF)
    # model_kp.setSeparating(SCIP_PARAMSETTING.OFF)
    # model_kp.setPresolve(SCIP_PARAMSETTING.OFF)
    # my_branchrule_test = MyBranching(model_kp)
    # model_kp.includeBranchrule(my_branchrule_test, "test branch", "test branching and probing and lp functions",priority=10000000, maxdepth=-1, maxbounddist=1)
    #
    # model_kp.optimize()
