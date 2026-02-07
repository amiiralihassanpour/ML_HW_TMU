from __future__ import annotations

from typing import Dict, List, Optional

from utils import validate_email, validate_grade, convert_to_letter_grade, convert_to_grade_points


StudentDict = Dict[str, Dict]   # students[student_id] = {"name":..., "email":..., "grades": {...}}
CourseDict = Dict[str, Dict]    # courses[course_code] = {"name":..., "credits":...}


class StudentManager:
    """Student CRUD operations."""

    def add_student(self, students: StudentDict, student_id: str, name: str, email: str) -> bool:
        if not student_id or not name or not email:
            print("Error: student_id, name, and email are required.")
            return False
        if student_id in students:
            print("Error: Student ID already exists!")
            return False
        if not validate_email(email):
            print("Error: Invalid email format!")
            return False

        students[student_id] = {"name": name.strip(), "email": email.strip(), "grades": {}}
        print(f"Student {name} added successfully!")
        return True

    def display_all_students(self, students: StudentDict) -> None:
        if not students:
            print("No students found.")
            return

        print("\n--- Students ---")
        for sid, info in students.items():
            print(f"{sid} | {info.get('name','')} | {info.get('email','')}")

    def search_student(self, students: StudentDict, student_id: str) -> Optional[Dict]:
        if student_id not in students:
            print("Student not found.")
            return None
        info = students[student_id]
        print(f"\nStudentID: {student_id}\nName: {info.get('name')}\nEmail: {info.get('email')}")
        return info

    def update_student(self, students: StudentDict, student_id: str, new_name: str, new_email: str) -> bool:
        if student_id not in students:
            print("Error: Student not found.")
            return False
        if new_email and not validate_email(new_email):
            print("Error: Invalid email format!")
            return False

        if new_name:
            students[student_id]["name"] = new_name.strip()
        if new_email:
            students[student_id]["email"] = new_email.strip()

        print("Student updated successfully!")
        return True


class CourseManager:
    """Course CRUD operations and metadata access."""

    def add_course(self, courses: CourseDict, course_code: str, course_name: str, credits: int) -> bool:
        if not course_code or not course_name:
            print("Error: course_code and course_name are required.")
            return False
        if course_code in courses:
            print("Error: Course code already exists!")
            return False
        try:
            credits = int(credits)
        except (TypeError, ValueError):
            print("Error: credits must be an integer.")
            return False
        if credits <= 0:
            print("Error: credits must be positive.")
            return False

        courses[course_code] = {"name": course_name.strip(), "credits": credits}
        print(f"Course {course_code} added successfully!")
        return True

    def display_all_courses(self, courses: CourseDict) -> None:
        if not courses:
            print("No courses found.")
            return

        print("\n--- Courses ---")
        for code, info in courses.items():
            print(f"{code} | {info.get('name','')} | credits={info.get('credits',0)}")

    def get_course_credits(self, courses: CourseDict, course_code: str) -> Optional[int]:
        if course_code not in courses:
            print("Error: Course not found.")
            return None
        return int(courses[course_code].get("credits", 0))


class GradeManager:
    """Enrollment, grade recording, GPA, and reports."""

    def enroll_student(self, students: StudentDict, courses: CourseDict, student_id: str, course_code: str) -> bool:
        if student_id not in students:
            print("Error: Student not found.")
            return False
        if course_code not in courses:
            print("Error: Course not found.")
            return False

        grades = students[student_id].setdefault("grades", {})
        if course_code not in grades:
            grades[course_code] = []
            print(f"Student {student_id} enrolled in {course_code}.")
        else:
            print("Student is already enrolled in this course.")
        return True

    def record_grade(self, students: StudentDict, courses: CourseDict, student_id: str, course_code: str, grade) -> bool:
        if student_id not in students:
            print("Error: Student not found.")
            return False
        if course_code not in courses:
            print("Error: Course not found.")
            return False
        if not validate_grade(grade):
            print("Error: Grade must be a number between 0 and 100.")
            return False

        grades = students[student_id].setdefault("grades", {})
        grades.setdefault(course_code, []).append(float(grade))
        print("Grade recorded successfully!")
        return True

    def calculate_course_average(self, students: StudentDict, course_code: str) -> Optional[float]:
        all_grades: List[float] = []
        for s in students.values():
            gdict = s.get("grades", {})
            if course_code in gdict and gdict[course_code]:
                all_grades.extend(gdict[course_code])

        if not all_grades:
            print("No grades found for this course.")
            return None
        return sum(all_grades) / len(all_grades)

    def calculate_student_gpa(self, students: StudentDict, courses: CourseDict, student_id: str) -> Optional[float]:
        if student_id not in students:
            print("Error: Student not found.")
            return None

        total_points = 0.0
        total_credits = 0

        for course_code, grades in students[student_id].get("grades", {}).items():
            if course_code not in courses or not grades:
                continue
            course_avg = sum(grades) / len(grades)
            letter = convert_to_letter_grade(course_avg)
            points = convert_to_grade_points(letter)
            credits = int(courses[course_code].get("credits", 0))

            total_points += points * credits
            total_credits += credits

        return (total_points / total_credits) if total_credits > 0 else 0.0

    def generate_student_report(self, students: StudentDict, courses: CourseDict, student_id: str) -> None:
        if student_id not in students:
            print("Error: Student not found.")
            return

        info = students[student_id]
        print("\n=== Student Report ===")
        print(f"StudentID: {student_id}")
        print(f"Name: {info.get('name')}")
        print(f"Email: {info.get('email')}")
        print("\nCourses & Grades:")

        grades_dict = info.get("grades", {})
        if not grades_dict:
            print("  (No enrollments/grades yet)")
        else:
            for code, grades in grades_dict.items():
                cname = courses.get(code, {}).get("name", "Unknown Course")
                avg = (sum(grades) / len(grades)) if grades else None
                avg_str = f"{avg:.2f}" if avg is not None else "N/A"
                print(f"  - {code} ({cname}): grades={grades} | avg={avg_str}")

        gpa = self.calculate_student_gpa(students, courses, student_id)
        if gpa is not None:
            print(f"\nGPA: {gpa:.2f}")

    def generate_course_report(self, students: StudentDict, courses: CourseDict, course_code: str) -> None:
        if course_code not in courses:
            print("Error: Course not found.")
            return

        cname = courses[course_code].get("name", "")
        print("\n=== Course Report ===")
        print(f"{course_code} - {cname}")

        enrolled = []
        for sid, sinfo in students.items():
            grades = sinfo.get("grades", {})
            if course_code in grades:
                enrolled.append((sid, sinfo.get("name", ""), grades[course_code]))

        if not enrolled:
            print("No enrolled students found.")
            return

        for sid, sname, sgrades in enrolled:
            avg = (sum(sgrades) / len(sgrades)) if sgrades else None
            avg_str = f"{avg:.2f}" if avg is not None else "N/A"
            print(f"  - {sid} | {sname} | grades={sgrades} | avg={avg_str}")

        course_avg = self.calculate_course_average(students, course_code)
        if course_avg is not None:
            print(f"\nCourse Average: {course_avg:.2f}")
