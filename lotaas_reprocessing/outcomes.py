"""Exit codes by which a pipeline stage reports an outcome other than failure.

No imports: the runner reads these on the node's own Python, outside the image.
"""

# pipeline_gpu: nothing is left to search once the beam is masked, or the beam is shorter than the dispersion
# sweep of the DM plan (a truncated recording; six beams of L632735/L632747 on 25 October 2026). The runner records the attempt as
# 'unusable' and the campaign excludes the beam instead of retrying it and asking a person.
# On 22 October 2026, 12 beams of 8 SAPs had 56-93% block-masked and all 648 channels persistent.
NO_USABLE_DATA = 3
