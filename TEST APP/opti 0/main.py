import numpy as np
import logging



class Vector:
    length_constraint= 0
    element_constraints= np.array([])
    dtype = float

    def __init__(self, values=None):
        if values is None:
            values = self.generate_default_values()

        if not all(isinstance(val, self.dtype) for val in values):
            logging.warning("Преобразование типов: входные данные не соответствуют типу %s", self.dtype)
                
        self.values = np.array(values, dtype=self.dtype)
        self.constrain_elements()

        if not self.check_constraints():
            logging.error("Вектор не соответствует заданным ограничениям при инициализации.")
            raise ValueError("Вектор не соответствует заданным ограничениям: %s" % self.values)

        logging.info("Вектор успешно инициализирован: %s", self.values)

    def generate_default_values(self):
        if Vector.length_constraint <= 0:
            raise ValueError("Length constraint must be greater than zero to generate default values.")

        if Vector.element_constraints.size > 0:
            lower_bounds, upper_bounds = Vector.element_constraints[:, 0], Vector.element_constraints[:, 1]
            return np.random.uniform(lower_bounds, upper_bounds, Vector.length_constraint)
        else:
            return np.zeros(self.length_constraint, dtype=Vector.dtype)

    def check_constraints(self):
        if self.length_constraint is not None and len(self.values) != self.length_constraint:
            logging.error("Длина вектора %d не соответствует ограничению %d", len(self.values), self.length_constraint)
            return False
        
        self.constrain_elements()
        return True

    def constrain_elements(self):
        """Приводит элементы вектора к ближайшим допустимым значениям в соответствии с ограничениями."""
        if len(self.element_constraints) > 0:
            lower_bounds, upper_bounds = self.element_constraints[:, 0], self.element_constraints[:, 1]

            if np.any(self.values < lower_bounds) or np.any(self.values > upper_bounds):
                logging.error(f"Элемент {self.values} меньше или больше ограничений {self.element_constraints}")

            self.values = np.clip(self.values, lower_bounds, upper_bounds)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value
        self.constrain_elements()

        if not self.check_constraints():
            logging.error("Изменение элемента приводит к несоответствию ограничениям: %s", self.values)
            raise ValueError("Изменение элемента приводит к несоответствию ограничениям.")

    def __iter__(self):
        return iter(self.values)

    def __add__(self, other):
        result = self._perform_operation(other, np.add)
        logging.info("Выполнено сложение двух векторов.")
        return result

    def __sub__(self, other):
        result = self._perform_operation(other, np.subtract)
        logging.info("Выполнено вычитание двух векторов.")
        return result

    def __mul__(self, other):
        result = self._perform_operation(other, np.multiply)
        logging.info("Выполнено умножение двух векторов.")
        return result

    def __truediv__(self, other):
        result = self._perform_operation(other, np.divide)
        logging.info("Выполнено деление двух векторов.")
        return result

    def _perform_operation(self, other, operation):
        if isinstance(other, Vector):
            new_values = operation(self.values, other.values)
        else:
            new_values = operation(self.values, other)
        result_vector = Vector(new_values)
        return result_vector

    def __repr__(self):
        return f"Вектор {self.values}"
    

# class Vector:
#     length_constraint= 0
#     element_constraints= np.array([])
#     dtype = float

#     def __init__(self, values=None):
#         if values is None:
#             values = self.generate_default_values()

#         if not all(isinstance(val, self.dtype) for val in values):
#             logging.warning("Преобразование типов: входные данные не соответствуют типу %s", self.dtype)
                
#         self.values = np.array(values, dtype=self.dtype)
#         self.constrain_elements()

#         if not self.check_constraints():
#             logging.error("Вектор не соответствует заданным ограничениям при инициализации.")
#             raise ValueError("Вектор не соответствует заданным ограничениям: %s" % self.values)

#         logging.info("Вектор успешно инициализирован: %s", self.values)

#     def generate_default_values(self):
#         if self.length_constraint <= 0:
#             raise ValueError("Length constraint must be greater than zero to generate default values.")

#         if self.element_constraints.size > 0:
#             lower_bounds, upper_bounds = self.element_constraints[:, 0], self.element_constraints[:, 1]
#             return np.random.uniform(lower_bounds, upper_bounds, self.length_constraint)
#         else:
#             return np.zeros(self.length_constraint, dtype=self.dtype)

#     def check_constraints(self):
#         if self.length_constraint is not None and len(self.values) != self.length_constraint:
#             logging.error("Длина вектора %d не соответствует ограничению %d", len(self.values), self.length_constraint)
#             return False
        
#         self.constrain_elements()
#         return True

#     def constrain_elements(self):
#         """Приводит элементы вектора к ближайшим допустимым значениям в соответствии с ограничениями."""
#         if len(self.element_constraints) > 0:
#             lower_bounds, upper_bounds = self.element_constraints[:, 0], self.element_constraints[:, 1]

#             if np.any(self.values < lower_bounds) or np.any(self.values > upper_bounds):
#                 logging.error(f"Элемент {self.values} меньше или больше ограничений {self.element_constraints}")

#             self.values = np.clip(self.values, lower_bounds, upper_bounds)

#     def __len__(self):
#         return len(self.values)

#     def __getitem__(self, index):
#         return self.values[index]

#     def __setitem__(self, index, value):
#         self.values[index] = value
#         self.constrain_elements()

#         if not self.check_constraints():
#             logging.error("Изменение элемента приводит к несоответствию ограничениям: %s", self.values)
#             raise ValueError("Изменение элемента приводит к несоответствию ограничениям.")

#     def __iter__(self):
#         return iter(self.values)

#     def __add__(self, other):
#         result = self._perform_operation(other, np.add)
#         logging.info("Выполнено сложение двух векторов.")
#         return result

#     def __sub__(self, other):
#         result = self._perform_operation(other, np.subtract)
#         logging.info("Выполнено вычитание двух векторов.")
#         return result

#     def __mul__(self, other):
#         result = self._perform_operation(other, np.multiply)
#         logging.info("Выполнено умножение двух векторов.")
#         return result

#     def __truediv__(self, other):
#         result = self._perform_operation(other, np.divide)
#         logging.info("Выполнено деление двух векторов.")
#         return result

#     def _perform_operation(self, other, operation):
#         if isinstance(other, Vector):
#             new_values = operation(self.values, other.values)
#         else:
#             new_values = operation(self.values, other)
#         result_vector = Vector(new_values)
#         return result_vector

#     def __repr__(self):
#         return f"Вектор {self.values}"

