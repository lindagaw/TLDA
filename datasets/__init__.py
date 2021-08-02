from .mnist import get_mnist
from .usps import get_usps
from .k_mnist import get_kmnist
from .svhn import get_svhn
from .office_home import get_office_home

__all__ = (get_usps, get_mnist, get_kmnist, get_svhn, get_office_home)
