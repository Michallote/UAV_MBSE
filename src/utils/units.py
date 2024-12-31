class Units:
    """
    A utility class for unit conversions to SI units.

    Attributes
    ----------
    mm : float
        Millimeters to meters conversion factor (1e-3).
    cm : float
        Centimeters to meters conversion factor (1e-2).
    m : float
        Meters to meters conversion factor (1).
    km : float
        Kilometers to meters conversion factor (1e3).
    inch : float
        Inches to meters conversion factor (0.0254).
    ft : float
        Feet to meters conversion factor (0.3048).
    yd : float
        Yards to meters conversion factor (0.9144).
    mile : float
        Miles to meters conversion factor (1609.34).
    mils : float
        Mils to meters conversion factor (0.0254 / 1000).

    g : float
        Grams to kilograms conversion factor (1e-3).
    kg : float
        Kilograms to kilograms conversion factor (1).
    tonne : float
        Tonnes to kilograms conversion factor (1e3).
    lb : float
        Pounds to kilograms conversion factor (0.453592).
    oz : float
        Ounces to kilograms conversion factor (0.0283495).

    s : float
        Seconds to seconds conversion factor (1).
    min : float
        Minutes to seconds conversion factor (60).
    hr : float
        Hours to seconds conversion factor (3600).
    day : float
        Days to seconds conversion factor (86400).

    Methods
    to_SI(value, unit)
        Converts a given value from the specified unit to its SI equivalent.
    """

    __slots__ = ()
    # Length units (meters)
    mm = 1e-3
    cm = 1e-2
    m = 1
    km = 1e3
    inch = 0.0254
    ft = 0.3048
    yd = 0.9144
    mile = 1609.34
    mils = 0.0254 / 1000

    # Mass units (kilograms)
    g = 1e-3
    kg = 1
    tonne = 1e3
    lb = 0.453592
    oz = 0.0283495

    # Time units (seconds)
    s = 1
    min = 60
    hr = 3600
    day = 86400

    @staticmethod
    def to_SI(value: float, unit: str) -> float:
        """Converts a given value from the specified unit to its SI equivalent.

        Parameters
        ----------
        value : float
            Value to convert.
        unit : str
            Unit to convert from.

        Returns
        -------
        float
            Value in SI units.
        """
        return value * getattr(Units, unit, 1)


if __name__ == "__main__":
    # Example usage
    print(Units.to_SI(10, "cm"))  # 0.1 (meters)
    print(Units.to_SI(5, "kg"))  # 5 (kilograms)
    print(Units.to_SI(2, "hr"))  # 7200 (seconds)
