import json
import csv
import os


def save_data_to_file(students, courses, filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {"students": students, "courses": courses},
                f,
                indent=2
            )
        print("Data saved.")
    except Exception as e:
        print("Error saving data:", e)


def load_data_from_file(filename):
    if not os.path.exists(filename):
        return {}, {}

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("students", {}), data.get("courses", {})
    except Exception as e:
        print("Error loading data:", e)
        return {}, {}


def export_grades_to_csv(students, courses, filename):
    try:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "StudentID", "Name", "Email",
                "CourseCode", "CourseName",
                "Credits", "Grades"
            ])

            for sid, sinfo in students.items():
                for ccode, grades in sinfo["grades"].items():
                    cinfo = courses.get(ccode, {})
                    writer.writerow([
                        sid,
                        sinfo["name"],
                        sinfo["email"],
                        ccode,
                        cinfo.get("name", ""),
                        cinfo.get("credits", ""),
                        grades
                    ])
        print("CSV exported.")
    except Exception as e:
        print("CSV export error:", e)
