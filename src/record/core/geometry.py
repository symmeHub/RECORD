"""
Quaternion utilities and lightweight geometry helpers.

This module provides a small Quaternion class and a few NumPy/Numba-accelerated
helpers to work with rotations:
- Hamilton product
- Conversion to Rodrigues vectors and rotation matrices
- Basic signed angle in the XY plane

Quaternion convention used throughout is [w, x, y, z] and unit quaternions are
expected for rotation-related operations unless stated otherwise.
"""

from abc import ABC
from typing import Dict, Any, List
import math
from math import atan2, pi

import numpy as np
from numba import njit

# import cv2


@njit
def hamilton_product(q1, q2):
    """
    Hamilton product of two quaternions.

    Parameters
    ----------
    q1 : numpy.ndarray
        First quaternion as array-like (4,), ordered as [w, x, y, z].
    q2 : numpy.ndarray
        Second quaternion as array-like (4,), ordered as [w, x, y, z].

    Returns
    -------
    numpy.ndarray
        Resulting quaternion (4,) in [w, x, y, z] order.
    """

    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2

    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2

    return np.array([a, b, c, d])


@njit
def quat_to_Rodrigues(q):
    """
    Convert a unit quaternion to a Rodrigues rotation vector.

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion (4,) in [w, x, y, z] order. Should be unit-norm.

    Returns
    -------
    numpy.ndarray
        Rodrigues rotation vector (3,) where direction is the rotation axis
        and norm is the rotation angle (radians).
    """

    angle = 2 * np.arccos(q[0])
    sin_angle = np.sqrt(1 - q[0] ** 2)
    axis = q[1:] / sin_angle if sin_angle != 0 else np.array([1, 0, 0])

    rodrigues = angle * axis
    return rodrigues


@njit
def quat_to_rotation_matrix(q):
    """
    Convert a unit quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion (4,) in [w, x, y, z] order. Should be unit-norm.

    Returns
    -------
    numpy.ndarray
        Rotation matrix (3, 3).
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    rotation_matrix = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    return rotation_matrix


@njit
def signed_angle(v1, v2):
    """
    Signed angle between two vectors in the XY plane.

    This computes atan2(cross, dot) using only x and y components, i.e.
    the signed angle from `v1` to `v2` in the XY plane.

    Parameters
    ----------
    v1 : numpy.ndarray
        First vector (at least 2 components; if 3, z is ignored here).
    v2 : numpy.ndarray
        Second vector (at least 2 components; if 3, z is ignored here).

    Returns
    -------
    float
        Signed angle in radians in (-pi, pi].
    """
    # Normaliser les vecteurs
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    # Calculer l'angle signé en radians
    angle = np.arctan2(cross_product, dot_product)
    return angle


class Quaternion:
    """
    Minimal quaternion class for 3D rotations.

    Notes
    -----
    - Internal storage and all APIs use [w, x, y, z] order.
    - Unless noted, methods assume the quaternion is unit-norm for proper
      rotation behavior.
    """

    def __init__(self, q):
        if len(q) != 4:
            raise ValueError("Input vector size error must be len(q)==4")
        self._q = np.array(q)

    def set_q(self, q: np.ndarray):
        if len(q) != 4:
            raise ValueError("Input vector size error must be len(q)==4")
        self._q = np.array(q)

    def get_q(self):
        return self._q

    def get_x(self):
        return self._q[1]

    def get_y(self):
        return self._q[2]

    def get_z(self):
        return self._q[3]

    def get_w(self):
        return self._q[0]

    q = property(get_q)
    x = property(get_x)
    y = property(get_y)
    z = property(get_z)
    w = property(get_w)

    @classmethod
    def from_rotvec(cls, rotvec):
        """
        Create a quaternion from a Rodrigues rotation vector.

        Parameters
        ----------
        rotvec : numpy.ndarray
            Rodrigues vector (3,). Direction is the rotation axis,
            norm is the angle in radians.

        Returns
        -------
        Quaternion
            Unit quaternion representing the same rotation.
        """
        angle = np.linalg.norm(rotvec)
        axis = rotvec / angle if angle != 0 else np.array([1, 0, 0])
        return cls.from_axis_angle(axis, angle)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """
        Create a quaternion from an axis-angle representation.

        Parameters
        ----------
        axis : numpy.ndarray
            Rotation axis (3,). Does not need to be normalized.
        angle : float
            Rotation angle in radians.

        Returns
        -------
        Quaternion
            Unit quaternion corresponding to the given axis-angle.
        """
        axis = axis / np.linalg.norm(axis)
        angle = angle / 2
        w = np.cos(angle)
        x, y, z = np.sin(angle) * axis
        return cls(np.array([w, x, y, z]))

    @classmethod
    def from_3Dvector(cls, u):
        """
        Create a pure-imaginary quaternion from a 3D vector.

        Parameters
        ----------
        u : numpy.ndarray
            Vector (3,).

        Returns
        -------
        Quaternion
            Quaternion [0, ux, uy, uz].
        """
        if len(u) == 3:
            return cls(np.array([0, u[0], u[1], u[2]]))
        else:
            raise ValueError("Input vector size error must be len(u)==3")

    def __getitem__(self, key):
        return self.q[key]

    def __repr__(self) -> str:
        return f"Quaternion(w:{self.w}, x:{self.x}, y:{self.y}, z:{self.z})"

    def __mul__(self, other):
        """
        Hamilton product with another quaternion.

        Parameters
        ----------
        other : Quaternion
            Right-hand quaternion operand.

        Returns
        -------
        Quaternion
            Product quaternion.
        """
        return Quaternion(hamilton_product(other.q, self.q))

    @property
    def q_conj(self):
        """
        Conjugate quaternion [w, -x, -y, -z].

        Returns
        -------
        Quaternion
        """
        return Quaternion(np.array([self.w, -self.x, -self.y, -self.z]))

    @property
    def norm(self):
        """
        Euclidean norm of the quaternion.
        """
        return np.linalg.norm(self.q)

    @property
    def norm_imag_part(self):
        """
        Euclidean norm of the imaginary part (x, y, z).
        """
        return np.linalg.norm(self.imag_part)

    @property
    def axis(self):
        """
        Unit rotation axis of the quaternion.

        Returns
        -------
        numpy.ndarray
            Unit vector (3,). Defaults to [1, 0, 0] for near-zero angles.
        """
        try:
            angle = self.angle
            if angle > 1e-6:
                x = self.x / np.sin(angle / 2)
                y = self.y / np.sin(angle / 2)
                z = self.z / np.sin(angle / 2)
                norm = np.linalg.norm(np.array([x, y, z]))
                return np.array([x, y, z]) / norm
            else:
                return np.array([1.0, 0.0, 0.0])
        except:
            return np.array([1.0, 0.0, 0.0])

    @property
    def q_inv(self):
        """
        Multiplicative inverse of the quaternion.

        Returns
        -------
        Quaternion
        """
        return Quaternion(
            np.array(
                [
                    self.q_conj.w / self.norm**2,
                    self.q_conj.x / self.norm**2,
                    self.q_conj.y / self.norm**2,
                    self.q_conj.z / self.norm**2,
                ]
            )
        )

    @property
    def imag_part(self):
        """
        Imaginary part of the quaternion.

        Returns
        -------
        numpy.ndarray
            Vector (x, y, z).
        """
        return np.array([self.x, self.y, self.z])

    @property
    def real_part(self):
        """
        Real part of the quaternion (w).
        """
        return self.w

    def _normalise(self):
        """
        In-place normalization to unit quaternion if non-zero.
        """
        if not self.is_unit():
            n = self.norm
            if n > 0:
                self.q = self.q / n

    def _wrap_angle(self, theta):
        """
        Wrap an angle to (-pi, pi], mapping odd multiples of pi to +pi.
        """
        result = ((theta + pi) % (2 * pi)) - pi
        if result == -pi:
            result = pi
        return result

    @property
    def angle(self):
        """
        Rotation angle in radians, in [0, 2*pi].
        """
        w = np.clip(self.w, -1.0, 1.0)
        return 2 * np.arccos(w)

    @property
    def angle_deg(self):
        """
        Rotation angle in degrees.
        """
        return np.degrees(self.angle)

    def conjugate(self):
        """
        Return the conjugate quaternion.
        """
        return self.q_conj

    @property
    def yaw(self):
        """Heading (yaw) in radians."""
        return math.atan2(
            2 * ((self.x * self.y) + (self.w * self.z)),
            self.w**2 + self.x**2 - self.y**2 - self.z**2,
        )

    @property
    def pitch(self):
        """Tilt (pitch) in radians."""
        return math.asin(2 * ((self.x * self.z) - (self.w * self.y)))

    @property
    def roll(self):
        """Bank (roll) in radians."""
        return math.atan2(
            2 * ((self.y * self.z) + (self.w * self.x)),
            self.w**2 - self.x**2 - self.y**2 + self.z**2,
        )

    @property
    def rotvec(self):
        """
        Rodrigues rotation vector (3,) equivalent to this quaternion.
        """
        return quat_to_Rodrigues(self.q)

    @property
    def rotation_matrix(self):
        """
        3x3 rotation matrix equivalent to this quaternion.
        """
        return quat_to_rotation_matrix(self.q)

    def rotate(self, u):
        """
        Rotate a vector or an array of vectors by this quaternion.

        Parameters
        ----------
        u : numpy.ndarray
            Either a single vector (3,) or an array of shape (n, 3).

        Returns
        -------
        numpy.ndarray
            Rotated vector with same shape as input.
        """
        if len(u.shape) == 1:
            if len(u) == 3:
                return ((self * Quaternion.from_3Dvector(u)) * self.q_inv).imag_part
            else:
                raise ValueError("Input vector size error must be len(u)==3")
        elif len(u.shape) == 2:
            if u.shape[1] == 3:
                # TODO: Vectorize this operation using jax or numpy broadcasting
                return np.array(
                    [
                        ((self * Quaternion.from_3Dvector(v)) * self.q_inv).imag_part
                        for v in u
                    ]
                )
            else:
                raise ValueError("Array shape error must be u.shape[1]==3")
        else:
            raise ValueError(
                f"Input must be a (3,) or (n,3) array. Got shape: {u.shape}"
            )

    @classmethod
    def compose_q02(cls, q01, q12):
        """
        Compose frame transforms to obtain q02 from q01 and q12.

        Parameters
        ----------
        q01 : Quaternion
            Rotation from frame 0 to frame 1.
        q12 : Quaternion
            Rotation from frame 1 to frame 2.

        Returns
        -------
        Quaternion
            Rotation from frame 0 to frame 2.
        """
        return q12 * q01

    @classmethod
    def compose_q12(cls, q02, q01):
        """
        Compose frame transforms to obtain q12 from q02 and q01.

        Parameters
        ----------
        q02 : Quaternion
            Rotation from frame 0 to frame 2.
        q01 : Quaternion
            Rotation from frame 0 to frame 1.

        Returns
        -------
        Quaternion
            Rotation from frame 1 to frame 2.
        """
        return q02 * q01.q_inv


# def apply_RT(P, R, T):
#     """
#     Applies RT transformation to 3D points P.
#     """
#     P = cv2.Rodrigues(R)[0].dot(P.T).T
#     P += T.T
#     return P


if __name__ == "__main__":
    v = np.array([0, 0, 1])
    q = Quaternion.from_axis_angle(axis=np.array([1, 0, 0]), angle=np.pi)
    vr = q.rotate(v)
