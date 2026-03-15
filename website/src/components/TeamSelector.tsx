import { useState } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Command, CommandInput, CommandList, CommandEmpty, CommandGroup, CommandItem } from '@/components/ui/command'
import { cn } from '@/lib/utils'
import { ChevronsUpDown, Check } from 'lucide-react'

interface TeamSelectorProps {
  teams: Record<string, string>
  selectedId: string | null
  onSelect: (id: string) => void
  placeholder?: string
  label?: string
}

export default function TeamSelector({
  teams,
  selectedId,
  onSelect,
  placeholder = 'Select a team...',
  label,
}: TeamSelectorProps) {
  const [open, setOpen] = useState(false)

  const teamEntries = Object.entries(teams).sort((a, b) => a[1].localeCompare(b[1]))

  return (
    <div className="flex flex-col gap-1.5">
      {label && (
        <label className="text-sm font-medium text-gray-600">{label}</label>
      )}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            role="combobox"
            aria-expanded={open}
            className={cn(
              'flex h-11 w-full items-center justify-between rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm',
              'ring-offset-white focus:outline-none focus:ring-2 focus:ring-navy-400 focus:ring-offset-2',
              'hover:bg-gray-50 transition-colors',
              !selectedId && 'text-gray-400'
            )}
          >
            {selectedId ? teams[selectedId] : placeholder}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </button>
        </PopoverTrigger>
        <PopoverContent className="w-[280px] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search teams..." />
            <CommandList>
              <CommandEmpty>No team found.</CommandEmpty>
              <CommandGroup>
                {teamEntries.map(([id, name]) => (
                  <CommandItem
                    key={id}
                    value={name}
                    onSelect={() => {
                      onSelect(id)
                      setOpen(false)
                    }}
                  >
                    <Check
                      className={cn(
                        'mr-2 h-4 w-4',
                        selectedId === id ? 'opacity-100' : 'opacity-0'
                      )}
                    />
                    {name}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  )
}
