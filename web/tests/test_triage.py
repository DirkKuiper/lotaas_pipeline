from euroflash import psrcat


def test_a_bright_pulsar_folded_at_a_fraction_of_its_harmonics_is_known():
    # Since the search sums 32 harmonics (16 October 2026) a fold's n-th harmonic, n up to 32, can be a pulsar's
    # m-th: B1541+09 (P 0.74845 s, DM 34.98) in its own beam folded at 23/4, 19/4, 31/2 ... of its period.
    pulsar = {'name': 'B1541+09', 'f0': 1 / 0.7484488274633482, 'dm': 34.9758}
    for period, relation in ((4.303207, '23/4'), (11.599899, '31/2'), (3.180633, '17/4')):
        assert psrcat.match(period, 35.2, [pulsar]) == (pulsar, relation)
    assert psrcat.match(0.7484488 * 20, 35.2, [pulsar])[1] == '20'          # whole multiples up to 32
    assert psrcat.match(0.7484488 * 33 / 4, 35.2, [pulsar]) is None         # beyond the 32nd harmonic
    assert psrcat.match(4.303207, 60.0, [pulsar]) is None                   # not at its DM
