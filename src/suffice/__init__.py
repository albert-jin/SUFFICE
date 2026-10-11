"""SUFFICE: query-only sufficiency and cost based retention selection."""
from .policy import RATIOS, admissible, select_ratios

__all__ = ['RATIOS', 'admissible', 'select_ratios']
