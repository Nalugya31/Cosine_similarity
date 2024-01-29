from faker import Faker
import random
import csv

faker = Faker()

# Predefine some choices
skills = ['Finance', 'Web Development', 'Graphic Design', 'Digital Marketing', 'Data Analysis', 'Nutrition', 'Nursing', 'Teaching', 'Engineering', 'Real Estate']
educations = ["Bachelor's in Business Administration", "Bachelor's in Computer Science", "Bachelor's in Design", "Bachelor's in Marketing", "Bachelor's in Data Science", "Bachelor's in Health Sciences", "Bachelor's in Nursing", "Bachelor's in Education", "Bachelor's in Engineering", "Bachelor's in Real Estate Management"]

with open('random_data.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Skill", "Education", "Experience"])

    for _ in range(6000):
        skill = random.choice(skills)
        if skill == 'Finance':
            education = "Bachelor's in Business Administration"
        elif skill == 'Web Development':
            education = "Bachelor's in Computer Science"
        elif skill == 'Graphic Design':
            education = "Bachelor's in Design"
        elif skill == 'Digital Marketing':
            education = "Bachelor's in Marketing"
        elif skill == 'Data Analysis':
            education = "Bachelor's in Data Science"
        elif skill == 'Nutrition':
            education = "Bachelor's in Health Sciences"
        elif skill == 'Nursing':
            education = "Bachelor's in Nursing"
        elif skill == 'Teaching':
            education = "Bachelor's in Education"
        elif skill == 'Engineering':
            education = "Bachelor's in Engineering"
        else: # Real Estate skill
            education = "Bachelor's in Real Estate Management"
            
        writer.writerow([
            faker.name(),
            skill,
            education,
            f"{random.randint(1,20)} years"     # generate random experience in years
        ])