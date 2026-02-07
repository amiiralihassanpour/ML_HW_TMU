from managers import StudentManager, CourseManager, GradeManager
from storage import save_data_to_file, load_data_from_file, export_grades_to_csv

students = {}
courses = {}

student_mgr = StudentManager()
course_mgr = CourseManager()
grade_mgr = GradeManager()

DATA_FILE = "HW1/data/data.json"
CSV_FILE = "HW1/data/export.csv"


# Required functions
def add_student(students, student_id, name, email):
    return student_mgr.add_student(students, student_id, name, email)


def display_all_students(students):
    student_mgr.display_all_students(students)


def search_student(students, student_id):
    return student_mgr.search_student(students, student_id)


def update_student(students, student_id, new_name, new_email):
    return student_mgr.update_student(students, student_id, new_name, new_email)


def add_course(courses, course_code, course_name, credits):
    return course_mgr.add_course(courses, course_code, course_name, credits)


def display_all_courses(courses):
    course_mgr.display_all_courses(courses)


def get_course_credits(courses, course_code):
    return course_mgr.get_course_credits(courses, course_code)


def enroll_student(students, student_id, course_code):
    return grade_mgr.enroll_student(students, courses, student_id, course_code)


def record_grade(students, student_id, course_code, grade):
    return grade_mgr.record_grade(students, courses, student_id, course_code, grade)


def calculate_course_average(students, course_code):
    return grade_mgr.calculate_course_average(students, course_code)


def calculate_student_gpa(students, courses, student_id):
    return grade_mgr.calculate_student_gpa(students, courses, student_id)


def display_menu():
    print("\n=== STUDENT GRADE MANAGEMENT SYSTEM ===")
    print("1. Add New Student")
    print("2. View All Students")
    print("3. Search Student")
    print("4. Update Student Information")
    print("5. Add New Course")
    print("6. View All Courses")
    print("7. Enroll Student in Course")
    print("8. Record Grade")
    print("9. Save Data")
    print("10. Load Data")
    print("11. Export CSV")
    print("0. Exit")


def main():
    global students, courses
    students, courses = load_data_from_file(DATA_FILE)

    while True:
        display_menu()
        choice = input("Choice: ")

        if choice == "1":
            add_student(
                students,
                input("ID: "),
                input("Name: "),
                input("Email: ")
            )

        elif choice == "2":
            display_all_students(students)

        elif choice == "3":
            search_student(students, input("ID: "))

        elif choice == "4":
            update_student(
                students,
                input("ID: "),
                input("New name: "),
                input("New email: ")
            )

        elif choice == "5":
            add_course(
                courses,
                input("Code: "),
                input("Name: "),
                input("Credits: ")
            )

        elif choice == "6":
            display_all_courses(courses)

        elif choice == "7":
            enroll_student(students, input("Student ID: "), input("Course code: "))

        elif choice == "8":
            record_grade(
                students,
                input("Student ID: "),
                input("Course code: "),
                input("Grade: ")
            )

        elif choice == "9":
            save_data_to_file(students, courses, DATA_FILE)

        elif choice == "10":
            students, courses = load_data_from_file(DATA_FILE)

        elif choice == "11":
            export_grades_to_csv(students, courses, CSV_FILE)

        elif choice == "0":
            save_data_to_file(students, courses, DATA_FILE)
            break


if __name__ == "__main__":
    main()
