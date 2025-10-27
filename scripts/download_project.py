import supervisely as sly

project_id = 4336
dest_dir = "project"
api = sly.Api()


sly.Project.download(
    api,
    project_id=project_id,
    dest_dir=dest_dir
)

print("done")