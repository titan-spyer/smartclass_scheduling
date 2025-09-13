#miniproject for timetable
from ortools.sat.python import cp_model
import itertools
from collections import defaultdict
import streamlit as st
import pandas as pd
import google.generativeai as genai

# Load API key from Streamlit secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, AttributeError):
    st.warning("`GOOGLE_API_KEY` not found in Streamlit secrets. AI features will be disabled. Please add it to your secrets in the deployment settings.", icon="⚠️")
    GOOGLE_API_KEY = None

teachers = ['Anand Sir', 'Diptimayi Mam', 'Adiyata Sir', 'Shakahambhri Mam', 'Sambeet Sir']
theory_subjects = ['MWE', 'DSP', 'AI/ML', 'VLSI', 'ISE' ]
lab_subjects = ['MWE lab', 'AI/ML lab', 'VLSI lab']
subjects = theory_subjects + lab_subjects
teacher_of = { 
    'MWE': 'Anand Sir',
    'DSP': 'Diptimayi Mam',
    'AI/ML': 'Shakahambhri Mam',
    'VLSI': 'Adiyata Sir',
    'ISE': 'Sambeet Sir',
    'MWE lab': 'Anand Sir',
    'AI/ML lab': 'Shakahambhri Mam',
    'VLSI lab': 'Adiyata Sir',
}
theory_batches = ['A', 'B']
lab_batches = ['A1', 'A2', 'B1', 'B2']
theory_batch_of_lab_batch = {
    'A1': 'A', 'A2': 'A',
    'B1': 'B', 'B2': 'B'
}
lab_batch_size = {'A1': 39, 'A2': 42, 'B1': 40, 'B2': 41}
theory_batches_size = {'A': 81, 'B': 81}
rooms = ['B307', 'B309']
room_cap = {'B307': 90, 'B309': 95}
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
daily_slots = ['9-10am', '10-11am', '11-12pm', '12-1pm', '2-3pm', '3-4pm', '4-5pm']
slot_times = {i: f'{d} {t}' for i, (d, t) in enumerate(itertools.product(days, daily_slots))}
slots = list(slot_times.keys())
slots_per_day = len(daily_slots)

valid_lab_start_slots = []
for sl in slots:
    day_index = sl % slots_per_day
    if day_index != (slots_per_day - 1) and day_index != 3:
        valid_lab_start_slots.append(sl)

nocp_theory = 3
nocp_lab = 1

model = cp_model.CpModel()


assign_theory = {}
for b in theory_batches:
    for s in theory_subjects:
        for sl in slots:
            for r in rooms:
                var = model.NewBoolVar(f'assign_theory_{b}_{s}_sl{sl}_{r}')
                assign_theory[(b, s, sl, r)] = var
                if theory_batches_size[b] > room_cap[r]:
                    model.Add(var == 0)

assign_lab = {}
for b in lab_batches:
    for s in lab_subjects:
        for sl in valid_lab_start_slots:
            for r in rooms:
                var = model.NewBoolVar(f'assign_lab_{b}_{s}_sl{sl}_{r}')
                assign_lab[(b, s, sl, r)] = var
                if lab_batch_size[b] > room_cap[r]:
                    model.Add(var == 0)

for b in theory_batches:
    for s in theory_subjects:
        model.Add(sum(assign_theory[(b, s, sl, r)] for sl in slots for r in rooms) == nocp_theory)

for b_th in theory_batches:
    for s_lab in lab_subjects:
        sub_batches = [b for b, th_b in theory_batch_of_lab_batch.items() if th_b == b_th]
        model.Add(sum(assign_lab[(b_lab, s_lab, sl, r)] 
                      for b_lab in sub_batches 
                      for sl in valid_lab_start_slots for r in rooms) == nocp_lab)

for b in theory_batches:
    for sl in slots:
        theory_class_in_slot = sum(assign_theory[b, s, sl, r] for s in theory_subjects for r in rooms)
        lab_classes_in_slot = []
        for b_lab in lab_batches:
            if theory_batch_of_lab_batch[b_lab] == b:
                lab_classes_in_slot.extend([assign_lab[b_lab, s, sl, r] for s in lab_subjects for r in rooms if sl in valid_lab_start_slots])
                if (sl - 1) in valid_lab_start_slots:
                    lab_classes_in_slot.extend([assign_lab[b_lab, s, sl - 1, r] for s in lab_subjects for r in rooms])
        model.Add(theory_class_in_slot + sum(lab_classes_in_slot) <= 1)

for sl in slots:
    for b_lab in lab_batches:
        lab_classes_in_slot = []
        lab_classes_in_slot.extend([assign_lab[b_lab, s, sl, r] for s in lab_subjects for r in rooms if sl in valid_lab_start_slots])
        if (sl - 1) in valid_lab_start_slots:
            lab_classes_in_slot.extend([assign_lab[b_lab, s, sl - 1, r] for s in lab_subjects for r in rooms])
        model.Add(sum(lab_classes_in_slot) <= 1)

for sl in slots:
    for r in rooms:
        theory_in_room = sum(assign_theory[b, s, sl, r] for b in theory_batches for s in theory_subjects)
        lab_in_room = []
        lab_in_room.extend([assign_lab[b, s, sl, r] for b in lab_batches for s in lab_subjects if sl in valid_lab_start_slots])
        if (sl - 1) in valid_lab_start_slots:
            lab_in_room.extend([assign_lab[b, s, sl - 1, r] for b in lab_batches for s in lab_subjects])
        model.Add(theory_in_room + sum(lab_in_room) <= 1)

for sl in slots:
    for t in teachers:
        theory_for_teacher = sum(assign_theory[b, s, sl, r] for b in theory_batches for s in theory_subjects if teacher_of[s] == t for r in rooms)
        lab_for_teacher = []
        lab_for_teacher.extend([assign_lab[b, s, sl, r] for b in lab_batches for s in lab_subjects if teacher_of[s] == t and sl in valid_lab_start_slots for r in rooms])
        if (sl - 1) in valid_lab_start_slots:
            lab_for_teacher.extend([assign_lab[b, s, sl - 1, r] for b in lab_batches for s in lab_subjects if teacher_of[s] == t for r in rooms])
        model.Add(theory_for_teacher + sum(lab_for_teacher) <= 1)

teacher_load = {}
max_load = model.NewIntVar(0, len(slots) * len(subjects), 'max_load')
for t in teachers:
    load = model.NewIntVar(0, len(slots) * len(subjects), f'load_{t}')
    teacher_load[t] = load
    theory_hours = sum(assign_theory[(b, s, sl, r)] for b in theory_batches for s in theory_subjects if teacher_of[s] == t for sl in slots for r in rooms)
    lab_hours = sum(assign_lab[(b, s, sl, r)] for b in lab_batches for s in lab_subjects if teacher_of[s] == t for sl in valid_lab_start_slots for r in rooms) * 2  # Labs are 2 hours
    model.Add(load == theory_hours + lab_hours)
    model.Add(load <= max_load)

student_daily_load = {}
max_student_daily_load = model.NewIntVar(0, slots_per_day, 'max_student_daily_load')
for b in theory_batches:
    for day_idx, d in enumerate(days):
        daily_slots_indices = [s for s in slots if s // slots_per_day == day_idx]
        
        theory_hours_on_day = sum(assign_theory[b, s, sl, r] 
                                  for s in theory_subjects 
                                  for sl in daily_slots_indices 
                                  for r in rooms)
        
        lab_hours_on_day_list = []
        for b_lab in lab_batches:
            if theory_batch_of_lab_batch[b_lab] == b:
                lab_hours_on_day_list.extend([assign_lab[b_lab, s, sl, r] * 2 
                                             for s in lab_subjects 
                                             for sl in daily_slots_indices if sl in valid_lab_start_slots 
                                             for r in rooms])
        
        load_var = model.NewIntVar(0, slots_per_day, f'student_load_{b}_day{day_idx}')
        model.Add(load_var == theory_hours_on_day + sum(lab_hours_on_day_list))
        student_daily_load[(b, day_idx)] = load_var
        model.Add(load_var <= max_student_daily_load)

teacher_daily_load = {}
max_teacher_daily_load = model.NewIntVar(0, slots_per_day, 'max_teacher_daily_load')
for t in teachers:
    for day_idx, d in enumerate(days):
        daily_slots_indices = [s for s in slots if s // slots_per_day == day_idx]

        theory_hours = sum(assign_theory[b, s, sl, r] for b in theory_batches for s in theory_subjects if teacher_of[s] == t for sl in daily_slots_indices for r in rooms)
        lab_hours_list = [assign_lab[b, s, sl, r] * 2 for b in lab_batches for s in lab_subjects if teacher_of[s] == t and sl in daily_slots_indices and sl in valid_lab_start_slots for r in rooms]
        
        load_var = model.NewIntVar(0, slots_per_day, f'teacher_load_{t}_day{day_idx}')
        model.Add(load_var == theory_hours + sum(lab_hours_list))
        teacher_daily_load[(t, day_idx)] = load_var
        model.Add(load_var <= max_teacher_daily_load)

model.Minimize(10 * max_student_daily_load + 10 * max_teacher_daily_load + max_load)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
solver.parameters.num_search_workers = 8
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    st.set_page_config(layout="wide")
    st.title('Generated University Timetable')
    st.success('Solution Found!')
    hectic_days_to_improve = []

    full_schedule = defaultdict(list)

    for b in theory_batches:
        for s in theory_subjects:
            for sl in slots:
                for r in rooms:
                    if solver.BooleanValue(assign_theory[(b, s, sl, r)]):
                        full_schedule[b].append((sl, s, r))

    for b_lab in lab_batches:
        b_th = theory_batch_of_lab_batch[b_lab]
        for s in lab_subjects:
            for sl in valid_lab_start_slots:
                for r in rooms:
                    if solver.BooleanValue(assign_lab[(b_lab, s, sl, r)]):
                        full_schedule[b_th].append((sl, f'{s} ({b_lab})', r))
                        full_schedule[b_th].append((sl + 1, f'{s} ({b_lab})', r))

    for b in theory_batches:
        st.header(f'Timetable for Batch {b}')

        df = pd.DataFrame(index=days, columns=daily_slots).fillna('')

        schedule = sorted(full_schedule[b])

        for sl, s, r in schedule:
            day_name, time_slot = slot_times[sl].split(' ', 1)
            teacher_key = s.split(' (')[0]
            teacher = teacher_of[teacher_key]
            cell_content = f"**{s}**<br>Room: {r}<br>By: {teacher}"
            df.loc[day_name, time_slot] += cell_content

        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)

    st.header('--- Workload Analysis ---')
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Weekly Teacher Workload')
        st.write(f'**Max weekly load:** {solver.Value(max_load)} hours')
        load_data = []
        for t in teachers:
            load_data.append({'Teacher': t, 'Load (hours)': solver.Value(teacher_load[t])})
        st.table(pd.DataFrame(load_data).set_index('Teacher'))

    with col2:
        st.subheader('Daily Load Limits')
        st.write(f"**Max daily student load:** {solver.Value(max_student_daily_load)} hours")
        st.write(f"**Max daily teacher load:** {solver.Value(max_teacher_daily_load)} hours")

        for (b, day_idx), load_var in student_daily_load.items():
            load_value = solver.Value(load_var)
            if load_value > 4:
                hectic_days_to_improve.append({
                    "batch": b, "day_index": day_idx, "day_name": days[day_idx],
                    "load": load_value, "load_var": load_var
                })

    if hectic_days_to_improve:
        st.header("💡 AI-Powered Schedule Improvement")
        most_hectic = max(hectic_days_to_improve, key=lambda x: x['load'])
        st.warning(f"Detected a potentially hectic day: **{most_hectic['day_name']}** for **Batch {most_hectic['batch']}** with **{most_hectic['load']} hours** of classes.")

        if st.button("Try to generate a less hectic schedule"):
            with st.spinner("Re-running solver with new constraints to improve the schedule..."):
                st.write(f"Adding constraint: Load for Batch {most_hectic['batch']} on {most_hectic['day_name']} must be less than {most_hectic['load']} hours.")
                model.Add(most_hectic['load_var'] < most_hectic['load'])

                new_status = solver.Solve(model)

                if new_status == cp_model.OPTIMAL or new_status == cp_model.FEASIBLE:
                    st.success("Found an improved schedule!")
                    st.header(f'Improved Timetable for Batch {most_hectic["batch"]}')

                    new_full_schedule = defaultdict(list)
                    for b in theory_batches:
                        for s in theory_subjects:
                            for sl in slots:
                                for r in rooms:
                                    if solver.BooleanValue(assign_theory[(b, s, sl, r)]):
                                        new_full_schedule[b].append((sl, s, r))
                    for b_lab in lab_batches:
                        b_th = theory_batch_of_lab_batch[b_lab]
                        for s in lab_subjects:
                            for sl in valid_lab_start_slots:
                                for r in rooms:
                                    if solver.BooleanValue(assign_lab[(b_lab, s, sl, r)]):
                                        new_full_schedule[b_th].append((sl, f'{s} ({b_lab})', r))
                                        new_full_schedule[b_th].append((sl + 1, f'{s} ({b_lab})', r))

                    df_improved = pd.DataFrame(index=days, columns=daily_slots).fillna('')
                    schedule_improved = sorted(new_full_schedule[most_hectic["batch"]])
                    for sl, s, r in schedule_improved:
                        day_name, time_slot = slot_times[sl].split(' ', 1)
                        teacher_key = s.split(' (')[0]
                        teacher = teacher_of[teacher_key]
                        cell_content = f"**{s}**<br>Room: {r}<br>By: {teacher}"
                        df_improved.loc[day_name, time_slot] += cell_content
                    st.markdown(df_improved.to_html(escape=False), unsafe_allow_html=True)
                else:
                    st.error("Could not find a better schedule. The current schedule is likely the most balanced possible under the existing rules.")

else:
    st.error('No solution found.')

def teacher_side_ui():
    st.header("👩‍🏫 Teacher's Assistant AI")
    st.info("This is a separate tool for teachers to ask questions. It does not affect the generated timetable.")

    if not (GOOGLE_API_KEY and genai.get_model('models/gemini-1.5-flash')):
        st.warning("The Teacher's Assistant AI is disabled because the Gemini API key is not configured correctly.", icon="⚠️")
        return

    prompt = st.text_area(
        "Ask the AI anything about your schedule, teaching strategies, or student engagement:",
        height=150,
        placeholder="e.g., 'Give me some ideas for an engaging first 10 minutes for my AI/ML class.' or 'What are common challenges when teaching VLSI?'"
    )

    if st.button("Ask AI Assistant"):
        if prompt:
            try:
                with st.spinner("The AI Assistant is thinking..."):
                    model_ai = genai.GenerativeModel('gemini-1.5-flash')
                    response = model_ai.generate_content(prompt)
                    st.markdown("### AI Assistant's Response:")
                    st.markdown(response.text)
            except Exception as e:
                st.error(f"An error occurred while contacting the Gemini API: {e}")
        else:
            st.warning("Please enter a question for the AI assistant.")

st.markdown("---")
teacher_side_ui()
