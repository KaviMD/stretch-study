#%%
import simplejson as json

#%%
with open("data/stretch-teleop.json") as f:
    data = json.load(f)
    
#%%

eventInfo = set()
eventName = set()

for user in data['users']:
    for event in data['users'][user]:
        event_data = data['users'][user][event]
        eventInfo.add(event_data['eventInfo'])
        eventName.add(event_data['eventName'])
# %%
print(eventInfo)
print(eventName)
# %%
