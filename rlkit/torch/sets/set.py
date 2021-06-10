import typing
import numpy as np


ExampleSet = typing.Dict[str, typing.Any]


class MathSet(object):
    def project(self, state):
        raise NotImplementedError()

    def distance_to_set(self, states):
        raise NotImplementedError()

    def describe(self):
        raise NotImplementedError()


class DebugSet(MathSet):
    def project(self, state):
        return state

    def distance_to_set(self, states):
        return np.zeros(states.shape[0])

    def describe(self):
        return 'debug'


class CustomSet(MathSet):
    def __init__(self, project_fn, distance_fn, description: str):
        self._project_fn = project_fn
        self._distance_fn = distance_fn
        self._description = description

    def project(self, state):
        return self._project_fn(state)

    def distance_to_set(self, states):
        return self._distance_fn(states)

    def describe(self):
        return self._description


class Set(typing.NamedTuple):
    description: MathSet
    example_dict: ExampleSet


class FixedPositionSet(MathSet):
    """Set where some elements have a set position."""
    def __init__(self, axis_idx_to_value):
        self._axis_idx_to_value = axis_idx_to_value

    def project(self, state):
        new_state = state.copy()
        for idx, value in self._axis_idx_to_value.items():
            new_state[idx] = value
        return new_state

    def distance_to_set(self, states):
        differences = []
        for idx, value in self._axis_idx_to_value.items():
            differences.append(states[..., idx] - value)
        delta_vectors = np.array(differences)
        return np.linalg.norm(delta_vectors, axis=0)

    def describe(self):
        return "distance_to_axes_" + "_".join(
            [
                str(idx)
                for idx in self._axis_idx_to_value
            ]
        )


class RelativePositionSet(MathSet):
    """
    Set of states where some elements have a position relative to another
    element.

    Usage:
    ```
    set = RelativePositionSet({
        0: 2,
        1: 3,
    })
    state = np.array([10, 11, 12, 13, 14])
    new_state = set.project(state)
    print(new_state)
    ```
    will output
    ```
    [12, 13, 12, 13, 14]
    ```
    """

    def __init__(
            self,
            a_axis_to_b_axis,
            offsets_from_b=None,
            max_value=None, min_value=None
    ):
        self.a_axis_to_b_axis = a_axis_to_b_axis
        if offsets_from_b is None:
            offsets_from_b = np.zeros(len(self.a_axis_to_b_axis))
        self.offsets_from_b = offsets_from_b
        self.max_value = max_value
        self.min_value = min_value
        if max_value is not None and min_value is not None:
            if max_value < min_value:
                raise ValueError("max cannot be less than min")

    def project(self, state):
        new_state = state.copy()
        for i, (a_i, b_i) in enumerate(self.a_axis_to_b_axis.items()):
            target_value = new_state[..., b_i] + self.offsets_from_b[i]
            if self.max_value is not None:
                to_move = target_value > self.max_value
                new_state[..., b_i] = (
                    (1 - to_move) * new_state[..., b_i]
                    + to_move * (self.max_value - self.offsets_from_b[i])
                )
                target_value = (
                    (1 - to_move) * target_value
                    + to_move * self.max_value
                )
            if self.min_value is not None:
                to_move = target_value < self.min_value
                new_state[..., b_i] = (
                        (1 - to_move) * new_state[..., b_i]
                        + to_move * (self.min_value - self.offsets_from_b[i])
                )
                target_value = (
                        (1 - to_move) * target_value
                        + to_move * self.min_value
                )
            new_state[..., a_i] = target_value
        return new_state

    def distance_to_set(self, states):
        projection = self.project(states)
        differences = states - projection
        return np.linalg.norm(differences, axis=-1)

    def describe(self):
        return "relative_distance_" + "_".join(
            [
                "{}to{}".format(a_i, b_i)
                for a_i, b_i in self.a_axis_to_b_axis.items()
            ]
        )


def sample_fixed_position_set(max_index, index=None):
    if index is None:
        index = np.random.randint(0, max_index//2)
    value = np.random.uniform(-4, 4, 1)
    return FixedPositionSet({index: value})


def sample_relative_position_set(max_index, index=None):
    if index is None:
        index = np.random.randint(0, max_index//2)
    value = np.random.uniform(-4, 4, 1)
    value2 = np.random.uniform(-4, 4, 1)
    return FixedPositionSet({
        2*index: value,
        2*index+1: value2,
    })