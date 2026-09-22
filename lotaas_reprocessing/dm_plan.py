"""Half-open DM grids without floating-point endpoint overlap."""
from decimal import Decimal, ROUND_CEILING


def dm_values(plan):
    low, high, step = (Decimal(str(plan[k])) for k in ('low_dm', 'high_dm', 'ddm'))
    if not all(v.is_finite() for v in (low, high, step)) or low < 0 or high <= low or step <= 0:
        raise ValueError('DM ranges require finite 0 <= low < high and a positive step')
    count = int(((high - low) / step).to_integral_value(rounding=ROUND_CEILING))
    return [float(low + i * step) for i in range(count)]


def dm_label(dm):
    # Retain historical labels such as DM10.0, allowing finer user-supplied grids.
    label = format(Decimal(str(dm)), 'f')
    if '.' in label:
        label = label.rstrip('0').rstrip('.')
    return label if '.' in label else label + '.0'
