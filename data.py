from faker import Faker
import random
import csv

faker = Faker()

# Predefine some choices
skills = {'Bachelors degree in Finance': ['Accounting', 'Economics', 'Data Analysis', 'Strategic Thinking', 'Business Awareness'],
          'Bachelors degree in Web Development': ['HTML/CSS', 'JavaScript', 'Backend Development', 'Frontend Development', 'Problem Solving'],
          'Bachelors degree in Graphic Design': ['Adobe Creative Suite', 'Layout Design', 'Typography', 'Visual Arts', 'Illustration'],
          'Bachelors degree in Digital Marketing': ['SEO/SEM', 'Content creation', 'Data analysis', 'Web design', 'Campaign Management'],
          'Bachelors degree in Data Analysis': ['Programming', 'Statistical modeling', 'Data manipulation', 'Database Administration', 'Presentation Skills'],
          'Bachelors degree in Nutrition': ['Food Assessment', 'Nutrition Therapy', 'Diet plan', 'Public Health', 'Molecular Biology'],
          'Bachelors degree in Nursing': ['Patient Care', 'Medical teamwork', 'Communication', 'Medicine Dosage Calculation', 'Time management'],
          'Bachelors degree in Teaching': ['Lesson Planning', 'Classroom Management', 'Education Psychology', 'Child Development', 'Teaching']}
educations = list(skills.keys())  # Update education list from the skills dictionary

with open('random_data.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Skill", "Education", "Experience"])

    for _ in range(6000):
        random_education = random.choice(educations)
        person_skills = random.sample(skills[random_education], 5)  # Sample 5 skills related to the education

        all_skills = ', '.join(person_skills)  # Join all skills with comma 

        writer.writerow([
            faker.name(),
            all_skills,
            random_education, 
            f"{random.randint(1, 20)} years"  # generate random experience in years
        ])