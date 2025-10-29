import supervisely as sly

project_id = 4336
dest_dir = "project"
api = sly.Api()


# sly.Project.download(
#     api,
#     project_id=project_id,
#     dest_dir=dest_dir
# )

print("done")

# project = sly.Project('project', mode=sly.OpenMode.READ)
sly.Project.to_segmentation_task(src_project_dir='project', dest_project_dir='semantic_project', segmentation_type="semantic")