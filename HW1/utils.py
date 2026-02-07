def validate_email(email: str) -> bool:
    """
    Validate email format.
    Conditions:
    - must contain '@'
    - must contain '.' after '@'
    Example: john@university.edu
    """
    if not isinstance(email, str):
        return False

    email = email.strip()

    if "@" not in email:
        return False

    local, _, domain = email.partition("@")

    if not local or not domain:
        return False

    if "." not in domain:
        return False

    return True


def validate_grade(grade) -> bool:
    """
    Grade must be a numeric value between 0 and 100.
    """
    try:
        g = float(grade)
    except (TypeError, ValueError):
        return False

    return 0.0 <= g <= 100.0


def convert_to_letter_grade(score: float) -> str:
    """
    Convert numeric score to letter grade.
    A: 90–100
    B: 80–89
    C: 70–79
    D: 60–69
    F: < 60
    """
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def convert_to_grade_points(letter: str) -> int:
    """
    Convert letter grade to GPA points.
    A = 4, B = 3, C = 2, D = 1, F = 0
    """
    grade_points = {
        "A": 4,
        "B": 3,
        "C": 2,
        "D": 1,
        "F": 0,
    }

    return grade_points.get(letter.upper(), 0)
