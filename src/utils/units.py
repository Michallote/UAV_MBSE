class Units:
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
    def to_SI(value, unit):
        return value * getattr(Units, unit, 1)


if __name__ == "__main__":
    # Example usage
    print(Units.to_SI(10, "cm"))  # 0.1 (meters)
    print(Units.to_SI(5, "kg"))  # 5 (kilograms)
    print(Units.to_SI(2, "hr"))  # 7200 (seconds)
