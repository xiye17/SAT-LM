import sys
import re
sys.path.append('.')

from collections import OrderedDict, namedtuple
from enum import Enum

TAB_STR = "    "
CHOICE_INDEXES = ["(A)", "(B)", "(C)", "(D)", "(E)"]

class CodeTranslator:
    class LineType(Enum):
        DECL = 1
        CONS = 2

    class ListValType(Enum):
        INT = 1
        ENUM = 2

    StdCodeLine = namedtuple("StdCodeLine", "line line_type")

    @staticmethod
    def translate_enum_sort_declaration(enum_sort_name, enum_sort_values):
        if all([x.isdigit() for x in enum_sort_values]):
            return [CodeTranslator.StdCodeLine(f"{enum_sort_name}_sort = IntSort()", CodeTranslator.LineType.DECL)]

        line = "{}, ({}) = EnumSort({}, [{}])".format(
            f"{enum_sort_name}_sort",
            ", ".join(enum_sort_values),
            f"'{enum_sort_name}'",
            ", ".join([f"'{x}'" for x in enum_sort_values])
        )
        return [CodeTranslator.StdCodeLine(line, CodeTranslator.LineType.DECL)]

    @staticmethod
    def translate_list_declaration(list_name, list_mebmers):
        line = "{} = [{}]".format(
            list_name,
            ", ".join(list_mebmers),
        )
        return [CodeTranslator.StdCodeLine(line, CodeTranslator.LineType.DECL)]

    @staticmethod
    def type_str_to_type_sort(arg):
        if arg == "bool":
            return "BoolSort()"
        elif arg == "int":
            return "IntSort()"
        else:
            return f"{arg}_sort"

    @staticmethod
    def translate_function_declaration(function_name, function_args):
        args = []
        for arg in function_args:
            args.append(CodeTranslator.type_str_to_type_sort(arg))

        line = "{} = Function('{}', {})".format(
            function_name,
            function_name,
            ", ".join(args),
        )
        return [CodeTranslator.StdCodeLine(line, CodeTranslator.LineType.DECL)]

    @staticmethod
    def extract_paired_token_index(statement, start_index, left_token, right_token):
        if statement[start_index] != left_token:
            raise RuntimeError("Invalid argument")

        level = 1
        for i in range(start_index + 1, len(statement)):
            if statement[i] == left_token:
                level += 1
            elif statement[i] == right_token:
                level -= 1
                if level == 0:
                    return i

    @staticmethod
    def extract_temperal_variable_name_and_scope(scope_contents):
        scope_fragments = [x.strip() for x in scope_contents.split(",")]
        return [x.split(":") for x in scope_fragments]

    @staticmethod
    def handle_count_function(statement):
        index = statement.find("Count(")
        content_start_index = index + len("Count")
        content_end_index = CodeTranslator.extract_paired_token_index(statement, content_start_index, "(", ")")
        count_arg_contents = statement[content_start_index + 1:content_end_index]
        scope_end_index = CodeTranslator.extract_paired_token_index(count_arg_contents, 0, "[", "]")
        scope_contents = count_arg_contents[1:scope_end_index]
        vars_and_scopes = CodeTranslator.extract_temperal_variable_name_and_scope(scope_contents)
        count_expr = count_arg_contents[scope_end_index + 1:].lstrip(", ")

        transformed_count_statement = "Sum([{} {}])".format(
            count_expr,
            " ".join([f"for {var} in {scope}" for var, scope in vars_and_scopes])
        )

        statement = statement[:index] + transformed_count_statement + statement[content_end_index + 1:]

        return statement

    @staticmethod
    def handle_distinct_function(statement):
        scoped_distinct_regex = r"Distinct\(\[[a-zA-Z0-9_]+:[a-zA-Z0-9_]"
        match = re.search(scoped_distinct_regex, statement)
        index = match.start()
        content_start_index = index + len("Distinct")
        content_end_index = CodeTranslator.extract_paired_token_index(statement, content_start_index, "(", ")")
        distinct_arg_contents = statement[content_start_index + 1:content_end_index]
        scope_end_index = CodeTranslator.extract_paired_token_index(distinct_arg_contents, 0, "[", "]")
        scope_contents = distinct_arg_contents[1:scope_end_index]
        vars_and_scopes = CodeTranslator.extract_temperal_variable_name_and_scope(scope_contents)
        distinct_expr = distinct_arg_contents[scope_end_index + 1:].lstrip(", ")
        assert len(vars_and_scopes) == 1
        transformed_distinct_statement = "Distinct([{} for {} in {}])".format(
            distinct_expr,
            vars_and_scopes[0][0],
            vars_and_scopes[0][1],
        )

        statement = statement[:index] + transformed_distinct_statement + statement[content_end_index + 1:]

        return statement
 
    @staticmethod
    def handle_quantifier_function(statement, scoped_list_to_type):
        scoped_quantifier_regex = r"(Exists|ForAll)\(\[([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)"
        match = re.search(scoped_quantifier_regex, statement)
        quant_name = match.group(1)

        index = match.start()
        content_start_index = index + len(quant_name)
        content_end_index = CodeTranslator.extract_paired_token_index(statement, content_start_index, "(", ")")
        quant_arg_contents = statement[content_start_index + 1:content_end_index]
        scope_end_index = CodeTranslator.extract_paired_token_index(quant_arg_contents, 0, "[", "]")
        scope_contents = quant_arg_contents[1:scope_end_index]
        vars_and_scopes = CodeTranslator.extract_temperal_variable_name_and_scope(scope_contents)
        quant_expr = quant_arg_contents[scope_end_index + 1:].lstrip(", ")

        var_need_declaration = []
        var_need_compresion = []
        for (var_name, var_scope) in vars_and_scopes:
            if var_scope in scoped_list_to_type:
                if scoped_list_to_type[var_scope] == CodeTranslator.ListValType.ENUM:
                    var_need_declaration.append((var_name, var_scope))
                else:
                    var_need_compresion.append((var_name, var_scope))
            else:
                assert var_scope in ["int", "bool"]
                var_need_declaration.append((var_name, var_scope))

        decl_lines = []
        std_scope = []
        if var_need_declaration:
            for (var_name, var_scope) in var_need_declaration:
                decl_lines.append(f"{var_name} = Const('{var_name}', {CodeTranslator.type_str_to_type_sort(var_scope)})")
                std_scope.append(var_name)
            std_constraint = "{}([{}], {})".format(quant_name, ", ".join(std_scope), quant_expr)
        else:
            std_constraint = quant_expr
    
        if var_need_compresion:
            logic_f = "And" if quant_name == "ForAll" else "Or"
            std_constraint = "{}([{} {}])".format(logic_f, std_constraint, " ".join([f"for {var_name} in {var_scope}" for (var_name, var_scope) in var_need_compresion]))

        std_constraint = statement[:index] + std_constraint + statement[content_end_index + 1:]

        return decl_lines, std_constraint

    @staticmethod
    def translate_constraint(constraint, scoped_list_to_type):
        # handle special operators into standard python operators
        while "Count(" in constraint:
            constraint = CodeTranslator.handle_count_function(constraint)

        scoped_distinct_regex = r"Distinct\(\[[a-zA-Z0-9_]+:[a-zA-Z0-9_]"
        # check if we can find scoped_distinct_regex in constraint
        while re.search(scoped_distinct_regex, constraint):
            constraint = CodeTranslator.handle_distinct_function(constraint)

        scoped_quantifier_regex = r"(Exists|ForAll)\(\[([a-zA-Z0-9_]+):([a-zA-Z0-9_]+)"
        all_decl_lines = []
        while re.search(scoped_quantifier_regex, constraint):
            decl_lines, constraint = CodeTranslator.handle_quantifier_function(constraint, scoped_list_to_type)
            all_decl_lines += decl_lines

        lines = [CodeTranslator.StdCodeLine(l, CodeTranslator.LineType.DECL) for l in all_decl_lines] + [CodeTranslator.StdCodeLine(constraint, CodeTranslator.LineType.CONS)]
        return lines

    @staticmethod
    def translate_option_verification(option_block, choice_name):
        lines = []
        lines.append("solver = Solver()")
        lines.append("solver.add(pre_conditions)")
        for l in option_block:
            lines.append("solver.add(Not({}))".format(l))
        lines.append("if solver.check() == unsat:")
        lines.append("\tprint('{}')".format(choice_name))
        return lines

    @staticmethod
    def assemble_standard_code(declaration_lines, pre_condidtion_lines, option_blocks):
        lines = []

        header_lines = [
            "from z3 import *", ""
        ]

        lines += header_lines
        lines += [x.line for x in declaration_lines]
        lines += [""]

        lines += ["pre_conditions = []"]
        for line in pre_condidtion_lines:
            if line.line_type == CodeTranslator.LineType.DECL:
                lines += [line.line]
            else:
                lines += ["pre_conditions.append({})".format(line.line)]
        lines += [""]

        function_lines = [
            "def is_valid(option_constraints):",
            TAB_STR + "solver = Solver()",
            TAB_STR + "solver.add(pre_conditions)",
            TAB_STR + "solver.add(Not(option_constraints))",
            TAB_STR + "return solver.check() == unsat",
            "",
            "def is_unsat(option_constraints):",
            TAB_STR + "solver = Solver()",
            TAB_STR + "solver.add(pre_conditions)",
            TAB_STR + "solver.add(option_constraints)",
            TAB_STR + "return solver.check() == unsat",
            "",
            "def is_sat(option_constraints):",
            TAB_STR + "solver = Solver()",
            TAB_STR + "solver.add(pre_conditions)",
            TAB_STR + "solver.add(option_constraints)",
            TAB_STR + "return solver.check() == sat",
            "",
            "def is_accurate_list(option_constraints):",
            TAB_STR + "return is_valid(Or(option_constraints)) and all([is_sat(c) for c in option_constraints])",
            "",
            "def is_exception(x):",
            TAB_STR + "return not x",
            ""
        ]


        lines += function_lines
        lines += [""]

        # handle option blocks
        for option_block, choice_name in zip(option_blocks, CHOICE_INDEXES):
            assert len([l for l in option_block if l.line_type == CodeTranslator.LineType.CONS]) == 1
            for line in option_block:
                if line.line_type == CodeTranslator.LineType.DECL:
                    lines += [line.line]
                else:
                    lines += [f"if {line.line}: print('{choice_name}')"]
        return "\n".join(lines)

class LSATSatProblem:
    def __init__(self, declared_enum_sorts, declared_lists, declared_functions, variable_constraints, sat_constraints, options):
        self.declared_enum_sorts = declared_enum_sorts
        self.declared_lists = declared_lists

        self.declared_functions = declared_functions
        self.variable_constraints = variable_constraints
        self.sat_constraints = sat_constraints
        self.options = options

    def __repr__(self):
        return f"LSATSatProblem:\n\tDeclared Enum Sorts: {self.declared_enum_sorts}\n\tDeclared Lists: {self.declared_lists}\n\tDeclared Functions: {self.declared_functions}\n\tConstraints: {self.sat_constraints}\n\tOptions: {self.options}"

    @classmethod
    def from_raw_statements(cls, raw_statements):
        lines = [x for x in raw_statements.splitlines() if x]
        assert "# declare variables" in lines
        assert "# constraints" in lines
        assert any([x.startswith("# the question asks") or x.startswith("# we check") for x in lines])
        decleration_start_index = lines.index("# declare variables")
        constraint_start_index = lines.index("# constraints")
        option_start_index = next(i for i, x in enumerate(lines) if x.startswith("# the question asks") or x.startswith("# we check"))
        declaration_statements = lines[decleration_start_index + 1:constraint_start_index]
        constraint_statements = lines[constraint_start_index + 1:option_start_index]
        option_statements = lines[option_start_index + 1:]

        (declared_enum_sorts, declared_lists, declared_functions, variable_constrants) = LSATSatProblem.parse_declaration_statements(declaration_statements)

        constraints = [x for x in constraint_statements if not x.startswith("#") and x]
        options = [x for x in option_statements if not x.startswith("#") and x]

        return cls(declared_enum_sorts, declared_lists, declared_functions, variable_constrants, constraints, options)

    @staticmethod    
    def parse_declaration_statements(declaration_statements):
        enum_sort_declarations = OrderedDict()
        function_declarations = OrderedDict()
        pure_declaration_statements = [x for x in declaration_statements if "Sort" in x or "Function" in x]
        variable_constrant_statements = [x for x in declaration_statements if not "Sort" in x and not "Function" in x]
        for s in pure_declaration_statements:
            if "EnumSort" in s:
                sort_name = s.split("=")[0].strip()
                sort_member_str = s.split("=")[1].strip()[len("EnumSort("):-1]
                sort_members = [x.strip() for x in sort_member_str[1:-1].split(",")]
                enum_sort_declarations[sort_name] = sort_members
            elif "Function" in s:
                function_name = s.split("=")[0].strip()
                if "->" in s and "[" not in s:
                    function_args_str = s.split("=")[1].strip()[len("Function("):]
                    function_args_str = function_args_str.replace("->", ",").replace("(", "").replace(")", "")
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
                elif "->" in s and "[" in s:
                    function_args_str = s.split("=")[1].strip()[len("Function("):-1]
                    function_args_str = function_args_str.replace("->", ",").replace("[", "").replace("]", "")
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
                else:
                    # legacy way
                    function_args_str = s.split("=")[1].strip()[len("Function("):-1]
                    function_args = [x.strip() for x in function_args_str.split(",")]
                    function_declarations[function_name] = function_args
            else:
                raise RuntimeError("Unknown declaration statement: {}".format(s))


        declared_enum_sorts = OrderedDict()
        declared_lists = OrderedDict()
        declared_functions = function_declarations
        already_declared = set()
        for name, members in enum_sort_declarations.items():
            # all contained by other enum sorts
            if all([x not in already_declared for x in members]):
                declared_enum_sorts[name] = members
                already_declared.update(members)
            declared_lists[name] = members

        return declared_enum_sorts, declared_lists, declared_functions, variable_constrant_statements

    def to_standard_code(self):
        declaration_lines = []
        # translate enum sorts
        for name, members in self.declared_enum_sorts.items():
            declaration_lines += CodeTranslator.translate_enum_sort_declaration(name, members)

        # translate lists
        for name, members in self.declared_lists.items():
            declaration_lines += CodeTranslator.translate_list_declaration(name, members)

        scoped_list_to_type = {}
        for name, members in self.declared_lists.items():
            if all(x.isdigit() for x in members):
                scoped_list_to_type[name] = CodeTranslator.ListValType.INT
            else:
                scoped_list_to_type[name] = CodeTranslator.ListValType.ENUM
        
        # translate functions
        for name, args in self.declared_functions.items():
            declaration_lines += CodeTranslator.translate_function_declaration(name, args)

        pre_condidtion_lines = []

        for constraint in self.variable_constraints:
            pre_condidtion_lines += CodeTranslator.translate_constraint(constraint, scoped_list_to_type)

        # additional function scope control
        for name, args in self.declared_functions.items():
            if args[-1] in scoped_list_to_type and scoped_list_to_type[args[-1]] == CodeTranslator.ListValType.INT:
                list_range = [int(x) for x in self.declared_lists[args[-1]]]
                assert list_range[-1] - list_range[0] == len(list_range) - 1
                scoped_vars = [x[0] + str(i) for i, x in enumerate(args[:-1])]
                func_call = f"{name}({', '.join(scoped_vars)})"

                additional_cons = "ForAll([{}], And({} <= {}, {} <= {}))".format(
                    ", ".join([f"{a}:{b}" for a, b in zip(scoped_vars, args[:-1])]),
                    list_range[0], func_call, func_call, list_range[-1]
                )
                pre_condidtion_lines += CodeTranslator.translate_constraint(additional_cons, scoped_list_to_type)


        for constraint in self.sat_constraints:
            pre_condidtion_lines += CodeTranslator.translate_constraint(constraint, scoped_list_to_type)

        # each block should express one option
        option_blocks = [CodeTranslator.translate_constraint(option, scoped_list_to_type) for option in self.options]

        return CodeTranslator.assemble_standard_code(declaration_lines, pre_condidtion_lines, option_blocks)
